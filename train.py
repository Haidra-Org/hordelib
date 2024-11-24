"""Train a predictive model from horde payload inputs to predict inference time.

Supports multi-processing, just run this multiple times and the processes will
automatically work together on the training. We are training with torch and searching
through network hyper parameters using Optuna.

Requires two input files (both exactly the same format) which can be created by enabling
the SAVE_KUDOS_TRAINING_DATA constant in the worker.
  - inference-time-data.json
  - inference-time-data-validation.json

The output is a series of model checkpoints, "kudos_models/kudos-X-n.ckpt" Where n is the
number of the trial and X is the study version. Once the best trial number is identified
simply select the appropriate file.

The stand-alone class in examples/kudos.py is the code to actually use the model.

Requires also a local mysql database named "optuna" and assumes it can connect
with user "root" password "root". Change to your needs.

For visualisation with optuna dashboard:
  optuna-dashboard mysql://root:root@localhost/optuna

This is a quick hack to assist with kudos calculation.
"""

import argparse
import enum
import json
import math
import os
import pickle
import random
import signal
import time
from collections import defaultdict
from typing import Any

import optuna
import optunahub
import torch
import torch.nn as nn
from loguru import logger
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.terminator import EMMREvaluator, MedianErrorEvaluator, Terminator, TerminatorCallback
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hordelib.horde import HordeLib

random.seed()


class PatienceState(enum.Enum):
    MAX_EPOCH = enum.auto()
    FIXED = enum.auto()
    SUGGEST = enum.auto()


# Number of trials to run.
# Each trial generates a new neural network topology with new hyper parameters and trains it.
NUMBER_OF_STUDY_TRIALS = 100

# Hyper parameter search bounds
NUM_EPOCHS = 2000
# Patience is an custom terminator that stops a training if no improvement has happened in this many epochs
USE_PATIENCE = PatienceState.FIXED
MIN_PATIENCE = 25
MAX_PATIENCE = 200
MIN_NUMBER_OF_EPOCHS = 50
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 5
MIN_NODES_IN_LAYER = 4
MAX_NODES_IN_LAYER = 128
MIN_LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1e-1
MIN_WEIGHT_DECAY = 1e-6
MAX_WEIGHT_DECAY = 1e-1
MIN_DATA_BATCH_SIZE = 32
MAX_DATA_BATCH_SIZE = 256
batch_start = int(math.ceil(math.log2(MIN_DATA_BATCH_SIZE)))
batch_end = int(math.floor(math.log2(MAX_DATA_BATCH_SIZE)))
batch_sizes = [2**i for i in range(batch_start, batch_end + 1)]

# The study sampler to use
USE_HEBO = False
if USE_HEBO:
    HEBOSampler = optunahub.load_module("samplers/hebo").HEBOSampler

    search_space = {
        "learning_rate": FloatDistribution(MIN_LEARNING_RATE, MAX_LEARNING_RATE, log=True),
        "weight_decay": FloatDistribution(MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, log=True),
        "batch_size": CategoricalDistribution(batch_sizes),
        "optimizer": CategoricalDistribution(["Adam", "RMSprop", "SGD"]),
        "hidden_layers": IntDistribution(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, log=True),
    }
    for i in range(MAX_HIDDEN_LAYERS):
        search_space[f"hidden_layer_{i}_size"] = IntDistribution(MIN_NODES_IN_LAYER, MAX_NODES_IN_LAYER, log=True)

    OPTUNA_SAMPLER = HEBOSampler(search_space)
else:
    OPTUNA_SAMPLER = optunahub.load_module("samplers/auto_sampler").AutoSampler()
# OPTUNA_SAMPLER = optuna.samplers.TPESampler(n_startup_trials=30, n_ei_candidates=30)
# OPTUNA_SAMPLER = optuna.samplers.NSGAIISampler()  # genetic algorithm

# We have the following inputs to our kudos calculation.
# The payload below is pruned from unused fields during tensor conversion
PAYLOAD_EXAMPLE = {
    "sdk_api_job_info": {
        "id_": "12ad89b7-ffaf-498e-9a6c-11690193dc23",
        "ids": ["12ad89b7-ffaf-498e-9a6c-11690193dc23"],
        "payload": {
            "sampler_name": "k_euler_a",
            "cfg_scale": 5.0,
            "denoising_strength": None,
            "seed": "2746011721",
            "height": 1024,
            "width": 1024,
            "seed_variation": None,
            "post_processing": [],
            "post_processing_order": "facefixers_first",
            "tiling": False,
            "hires_fix": False,
            "hires_fix_denoising_strength": None,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "facefixer_strength": None,
            "loras": None,
            "tis": None,
            "extra_texts": None,
            "workflow": None,
            "transparent": False,
            "use_nsfw_censor": False,
            "ddim_steps": 20,
            "n_iter": 1,
            "scheduler": "karras",
            "lora_count": 0,
            "ti_count": 0,
        },
        "model": "Unstable Diffusers XL",
        "source_processing": "img2img",
        "model_baseline": "stable_diffusion_xl",
        "extra_source_images_count": 0,
        "extra_source_images_combined_size": 0,
        "source_image_size": 0,
        "source_mask_size": 0,
    },
    "state": "ok",
    "censored": False,
    "time_popped": 1730365238.8009083,
    "time_submitted": 1730365253.0033202,
    "time_to_generate": 6.871337175369263,
    "time_to_download_aux_models": None,
}


KNOWN_POST_PROCESSORS = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "NMKD_Siax",
    "4x_AnimeSharp",
    "strip_background",
    "GFPGAN",
    "CodeFormers",
]
KNOWN_SCHEDULERS = [
    "simple",
    "karras",
]
KNOWN_SCHEDULERS.sort()
KNOWN_SAMPLERS = sorted(set(HordeLib.SAMPLERS_MAP.keys()))
KNOWN_CONTROL_TYPES = list(set(HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys()))
KNOWN_CONTROL_TYPES.append("None")
KNOWN_CONTROL_TYPES.sort()
KNOWN_SOURCE_PROCESSING = HordeLib.SOURCE_IMAGE_PROCESSING_OPTIONS[:]
KNOWN_SOURCE_PROCESSING.append("txt2img")
KNOWN_SOURCE_PROCESSING.sort()
KNOWN_MODEL_BASELINES = [
    "stable_diffusion_1",
    "stable_diffusion_2",
    "stable_diffusion_xl",
    "stable_cascade",
    "flux_1",
]
KNOWN_MODEL_BASELINES.sort()
KNOWN_WORKFLOWS = [
    "autodetect",
    "qr_code",
]
KNOWN_WORKFLOWS.sort()


def parse_args():
    parser = argparse.ArgumentParser(description="ML Training Script with configurable parameters")

    # Training control
    parser.add_argument("-e", "--enable-training", action="store_true", default=False, help="Enable training mode")
    parser.add_argument(
        "-a",
        "--analyse",
        action="store_true",
        default=False,
        help="When True will analyse and report which values in the bad predictions are the most common",
    )

    # Test mode
    parser.add_argument("-t", "--test-model", type=str, help="Path to model file for testing one by one")

    # Database configuration
    parser.add_argument(
        "--db-path",
        type=str,
        default="optuna_studies.db",
        help="Path to SQLite database file for Optuna",
    )

    # Data paths
    parser.add_argument(
        "--training-data",
        type=str,
        default="./inference-time-data.analyzed.json",
        help="Path to training data file",
    )

    parser.add_argument(
        "--validation-data",
        type=str,
        default="./inference-time-data-validation.analyzed.json",
        help="Path to validation data file",
    )

    parser.add_argument(
        "-p",
        "--progress-bars",
        action="store_true",
        default=False,
        help="Enable progress bars for epoch and trial progress",
    )

    # Study parameters
    parser.add_argument("--study-trials", type=int, default=2000, help="Number of trials to run")

    parser.add_argument("-v", "--study-version", type=str, default="v25", help="Version number of the study")

    return parser.parse_args()


class AbortTrial(Exception):
    pass


def signal_handler(sig, frame):
    raise AbortTrial


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def payload_to_time(model: nn.Module, payload: dict) -> float:
    """Return the predicted time in seconds for a given horde payload."""
    inputs = KudosDataset.payload_to_tensor(payload).squeeze()
    with torch.no_grad():
        output: torch.Tensor = model(inputs)
    return round(float(output.item()), 2)


# This is how to load the model required above
def load_model(model_filename) -> nn.Module:
    with open(model_filename, "rb") as infile:
        return pickle.load(infile)


class PercentageLoss(torch.nn.Module):
    """Torch module to calculate the percentage loss between two tensors."""

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """Calculate the percentage loss between the predicted and actual time.
        Args:
            predicted (torch.Tensor): The predicted time in seconds
            actual (torch.Tensor): The actual time in seconds

        Returns:
            torch.Tensor: The percentage loss
        """
        diff = torch.abs(actual - predicted)
        max_val = torch.max(actual, predicted)
        # We make it an order of magnitude higher, so that it appears clearer on the graphs
        return (diff / max_val).mean() * 100


def flatten_dict(d: dict, parent_key: str = "") -> dict[str, Any]:
    """
    Flatten nested dictionaries, keeping only specific keys.
    """
    ALLOWED_KEYS = {
        "height",
        "width",
        "ddim_steps",
        "cfg_scale",
        "denoising_strength",
        "clip_skip",
        "control_strength",
        "facefixer_strength",
        "lora_count",
        "ti_count",
        "extra_source_images_count",
        "extra_source_images_combined_size",
        "source_image_size",
        "source_mask_size",
        "hires_fix",
        "hires_fix_denoising_strength",
        "image_is_control",
        "return_control_map",
        "transparent",
        "tiling",
        "post_processing_order",
    }

    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        elif k in ALLOWED_KEYS:
            items.append((k, v))
    return dict(items)


def analyze_dict_similarities(
    dict_list: list[dict],
    float_tolerance: float = 0.1,
) -> dict[str, list[tuple[Any, float]]]:
    """
    Analyzes a list of dictionaries to find common or similar values across all dictionaries.
    """

    def are_values_similar(val1: Any, val2: Any) -> bool:
        """Check if two values are similar based on their type."""
        if type(val1) != type(val2):
            return False

        if isinstance(val1, int | float) and isinstance(val2, int | float):
            if val1 == 0 or val2 == 0:
                return abs(val1 - val2) < float_tolerance
            diff = abs(val1 - val2) / max(abs(val1), abs(val2))
            return diff <= float_tolerance

        return val1 == val2

    # Flatten all dictionaries
    flattened_dicts = [flatten_dict(d) for d in dict_list]

    # Get all unique keys
    all_keys = set().union(*[d.keys() for d in flattened_dicts])

    # Initialize results
    results = {}

    # Analyze each key
    for key in all_keys:
        values = [d.get(key) for d in flattened_dicts if key in d]
        if not values:
            continue

        value_groups: dict[str, float] = defaultdict(int)
        processed_values = set()

        for i, val1 in enumerate(values):
            if val1 is None or str(val1) in processed_values:
                continue

            count = 1
            for val2 in values[i + 1 :]:
                if are_values_similar(val1, val2):
                    count += 1

            if count > 1:
                value_groups[str(val1)] = count / len(values)
                processed_values.add(str(val1))

        if value_groups:
            sorted_items = sorted(value_groups.items(), key=lambda x: x[1], reverse=True)
            results[key] = list(sorted_items)

    return results


def print_similarity_analysis(results: dict[str, list[tuple[Any, float]]], min_frequency: float = 0.5) -> None:
    """
    Pretty prints the similarity analysis results.
    """
    print("\nSimilarity Analysis Results:")
    print("=" * 80)

    for key, values in sorted(results.items()):
        filtered_values = [(val, freq) for val, freq in values if freq >= min_frequency]
        if filtered_values:
            print(f"\nKey: {key}")
            print("-" * 40)
            for value, frequency in filtered_values:
                percentage = frequency * 100
                print(f"Value: {value:20} Frequency: {percentage:.1f}%")


# This is just an helper for walking through the validation dataset one line at a time
# and using the methods above to calculate an overall average percentage accuracy
def test_one_by_one(model_filename):
    dataset = []
    with open(VALIDATION_DATA_FILENAME) as infile:
        d = json.load(infile)
        for p in d:
            if p["time_to_generate"] is None:
                continue
            if p["state"] == "faulted":
                continue
            # We avoid some very rare types of jobs for now
            if p["sdk_api_job_info"]["payload"].get("transparent", False) is True:
                continue
            if p["sdk_api_job_info"]["payload"].get("tiling", False) is True:
                continue
            if p["sdk_api_job_info"]["payload"].get("return_control_map", False) is True:
                continue
            if p["sdk_api_job_info"]["payload"].get("workflow", "autodetect") not in [None, "autodetect"]:
                continue
            # We assume 5+ minutes for a gen on <10 steps is an extreme outlier we won't use
            if p["time_to_generate"] > 300 and p["sdk_api_job_info"]["payload"]["ddim_steps"] < 10:
                continue
            dataset.append(p)

    model = load_model(model_filename)

    perc = []
    total_job_time = 0
    total_time = 0
    bad_predictions = []
    within_1_second_count = 0
    within_2_seconds_count = 0
    within_3_seconds_count = 0
    within_5_seconds_count = 0
    within_10_seconds_count = 0

    total_number_of_jobs = len(dataset)

    for data in dataset:
        model_time = time.perf_counter()
        predicted = payload_to_time(model, data)
        total_time += time.perf_counter() - model_time
        actual = round(data["time_to_generate"], 2)
        total_job_time += data["time_to_generate"]

        diff = abs(actual - predicted)
        max_val = max(actual, predicted)
        percentage_accuracy = (1 - diff / max_val) * 100
        perc.append(percentage_accuracy)
        if diff <= 1:
            within_1_second_count += 1
        if diff <= 2:
            within_2_seconds_count += 1
        if diff <= 3:
            within_3_seconds_count += 1
        if diff <= 5:
            within_5_seconds_count += 1
        if diff <= 10:
            within_10_seconds_count += 1
        if percentage_accuracy < 60:
            bad_predictions.append(data)

    avg_perc = round(sum(perc) / len(perc), 1)
    print(f"Average kudos calculation time {round((total_time*1000000)/len(perc))} micro-seconds")
    print(f"Average actual job time in the dataset {round(total_job_time/len(perc), 2)} seconds")
    print(f"Average accuracy = {avg_perc}%")
    print(f"% predictions within 1 second of actual: {round((within_1_second_count/total_number_of_jobs)*100, 2)}%")
    print(f"% predictions within 2 seconds of actual: {round((within_2_seconds_count/total_number_of_jobs)*100, 2)}%")
    print(f"% predictions within 3 seconds of actual: {round((within_3_seconds_count/total_number_of_jobs)*100, 2)}%")
    print(f"% predictions within 5 seconds of actual: {round((within_5_seconds_count/total_number_of_jobs)*100, 2)}%")
    print(
        f"% predictions within 10 seconds of actual: {round((within_10_seconds_count/total_number_of_jobs)*100, 2)}%",
    )

    return bad_predictions


class KudosDataset(Dataset):
    def __init__(self, filename):
        logger.debug(f"Loading dataset from {filename}")
        self.data = []
        self.labels = []

        with open(filename) as infile:
            payload_list = json.load(infile)

            for payload in payload_list:
                if payload["time_to_generate"] is None:
                    continue
                if payload["state"] == "faulted":
                    continue
                # We avoid some very rare types of jobs for now
                if payload["sdk_api_job_info"]["payload"].get("transparent", False) is True:
                    continue
                if payload["sdk_api_job_info"]["payload"].get("tiling", False) is True:
                    continue
                if payload["sdk_api_job_info"]["payload"].get("return_control_map", False) is True:
                    continue
                if payload["sdk_api_job_info"]["payload"].get("workflow", "autodetect") not in [None, "autodetect"]:
                    continue
                # We assume 5+ minutes for a gen on <10 steps is an extreme outlier we won't use
                if payload["time_to_generate"] > 300 and payload["sdk_api_job_info"]["payload"]["ddim_steps"] < 10:
                    continue
                self.data.append(KudosDataset.payload_to_tensor(payload)[0])
                self.labels.append(payload["time_to_generate"])
        self.labels = torch.tensor(self.labels).float()
        self.mixed_data = torch.stack(self.data)
        logger.debug(f"Loaded {len(self.data)} samples")

    @classmethod
    def payload_to_tensor(cls, payload: dict) -> torch.Tensor:
        payload = payload["sdk_api_job_info"]
        p = payload["payload"]
        data = []
        data_samplers = []
        data_control_types = []
        data_source_processing_types = []
        data_model_baseline = []
        data_post_processors = []
        data_schedulers = []
        data_workflows = []
        data.append(
            [
                p["height"] * p["width"] / (1024 * 1024),
                p["ddim_steps"] / 100,
                p.get("denoising_strength", 1.0) if p.get("denoising_strength", 1.0) is not None else 1.0,
                p.get("control_strength", 1.0) if p.get("control_strength", 1.0) is not None else 1.0,
                p.get("facefixer_strength", 1.0) if p.get("facefixer_strength", 1.0) is not None else 1.0,
                p.get("lora_count", 0.0) / 5,
                p.get("ti_count", 0.0) / 10,
                p.get("extra_source_images_count", 0.0) / 5,
                p.get("extra_source_images_combined_size", 0.0) / 100_000,
                p.get("source_image_size", 0.0) / 100_000,
                p.get("source_mask_size", 0.0) / 100_000,
                1.0 if p.get("hires_fix", True) else 0.0,
                1.0 if p.get("image_is_control", True) else 0.0,
                1.0 if p.get("return_control_map", True) else 0.0,
                1.0 if p.get("transparent", True) else 0.0,
                1.0 if p.get("tiling", True) else 0.0,
                1.0 if p.get("post_processing_order", "facefixers_first") == "facefixers_first" else 0.0,
            ],
        )
        data_model_baseline.append(
            payload["model_baseline"] if payload["model_baseline"] in KNOWN_MODEL_BASELINES else "stable_diffusion_xl",
        )
        data_schedulers.append(p["scheduler"])
        data_samplers.append(p["sampler_name"] if p["sampler_name"] in KNOWN_SAMPLERS else "k_euler")
        data_control_types.append(
            p.get("control_type", "None") if p.get("control_type", "None") is not None else "None",
        )
        data_source_processing_types.append(payload.get("source_processing", "txt2img"))
        data_post_processors = p.get("post_processing", [])[:]
        data_workflows.append(
            p.get("workflow", "autodetect") if p.get("workflow", "autodetect") is not None else "autodetect",
        )

        _data_floats = torch.tensor(data).float()
        _data_model_baselines = cls.one_hot_encode(data_model_baseline, KNOWN_MODEL_BASELINES)
        _data_samplers = cls.one_hot_encode(data_samplers, KNOWN_SAMPLERS)
        _data_schedulers = cls.one_hot_encode(data_schedulers, KNOWN_SCHEDULERS)
        _data_control_types = cls.one_hot_encode(data_control_types, KNOWN_CONTROL_TYPES)
        _data_source_processing_types = cls.one_hot_encode(data_source_processing_types, KNOWN_SOURCE_PROCESSING)
        _data_workflows = cls.one_hot_encode(data_workflows, KNOWN_WORKFLOWS)
        _data_post_processors = cls.one_hot_encode_combined(data_post_processors, KNOWN_POST_PROCESSORS)
        return torch.cat(
            (
                _data_floats,
                _data_model_baselines,
                _data_samplers,
                _data_schedulers,
                _data_control_types,
                _data_source_processing_types,
                _data_post_processors,
                _data_workflows,
            ),
            dim=1,
        )

    @classmethod
    def one_hot_encode(cls, strings, unique_strings):
        one_hot = torch.zeros(len(strings), len(unique_strings))
        for i, string in enumerate(strings):
            one_hot[i, unique_strings.index(string)] = 1
        return one_hot

    @classmethod
    def one_hot_encode_combined(cls, strings, unique_strings):
        one_hot = torch.zeros(len(strings), len(unique_strings))
        for i, string in enumerate(strings):
            one_hot[i, unique_strings.index(string)] = 1

        return torch.sum(one_hot, dim=0, keepdim=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.mixed_data[idx], self.labels[idx]


def create_sequential_model(
    trial: optuna.Trial,
    layer_sizes: list[int],
    input_size: int,
    output_size: int = 1,
) -> nn.Sequential:
    logger.debug(
        f"Creating model with input size {input_size}, output size {output_size}, and layer sizes {layer_sizes}",
    )
    # Define the layer sizes
    layer_sizes = [input_size] + layer_sizes + [output_size]

    # Create the layers and activation functions
    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())  # Use ReLU activation for all layers except the last one
            # Add a dropout layer
            if i > 0:
                if USE_HEBO:
                    drop = 0.08
                else:
                    drop = trial.suggest_float(f"dropout_l{i}", 0.05, 0.2, log=True)
                layers.append(nn.Dropout(drop))

    # Create the nn.Sequential model
    return nn.Sequential(*layers)


def objective(trial: optuna.Trial) -> float:
    logger.debug(f"Starting trial {trial.number}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trial.set_user_attr("name", "predict_kudos")

    # Network topology
    input_size = len(KudosDataset.payload_to_tensor(PAYLOAD_EXAMPLE)[0])
    num_hidden_layers = trial.suggest_int("hidden_layers", MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS, log=True)
    layers = []
    for i in range(num_hidden_layers):
        layers.append(
            trial.suggest_int(f"hidden_layer_{i}_size", MIN_NODES_IN_LAYER, MAX_NODES_IN_LAYER, log=True),
        )
    output_size = 1  # we want just the predicted time in seconds

    # Create the network
    model = create_sequential_model(trial, layers, input_size, output_size).to(device)

    # Optimiser
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("learning_rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE, log=True)
    weight_decay = trial.suggest_float("weight_decay", MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, log=True)

    optimizer: optim.Optimizer | None = None

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer is None:
        raise Exception("Unknown optimizer")

    # Load training dataset
    train_dataset = KudosDataset(TRAINING_DATA_FILENAME)
    # suggest_categorical returns a numpy.int64 and that
    # causes an exception when using HEBO, so we convert to simple int.
    batch = int(trial.suggest_categorical("batch_size", batch_sizes))
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # Load the validation dataset
    validate_dataset = KudosDataset(VALIDATION_DATA_FILENAME)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=True)

    if validate_loader is None:
        raise Exception("No validation data")
    # Loss function
    # criterion = nn.HuberLoss()
    criterion = PercentageLoss()

    total_loss = 0.0
    best_epoch = 0
    best_loss = float("inf")
    best_state_dict = None

    if USE_PATIENCE == PatienceState.SUGGEST:
        patience = trial.suggest_int("patience", MIN_PATIENCE, MAX_PATIENCE)
    elif USE_PATIENCE == PatienceState.FIXED:
        patience = MAX_PATIENCE
    epochs_since_best = 0

    pbar: range | tqdm

    if ENABLE_PROGRESS_BARS:
        pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")
    else:
        pbar = range(NUM_EPOCHS)

    logger.debug(f"Starting training for {NUM_EPOCHS} epochs")

    for epoch in pbar:
        # Train the model
        model.train()
        data: torch.Tensor
        labels: torch.Tensor
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            outputs = model(data)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, labels in validate_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                total_loss += float(loss)

        total_loss /= len(validate_loader)
        total_loss = round(float(total_loss), 4)
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
            epochs_since_best = 0
        else:
            epochs_since_best = epoch - best_epoch
            if USE_PATIENCE != PatienceState.MAX_EPOCH and epochs_since_best >= patience:
                # Stop early, no improvement in awhile
                break

        info_str = (
            f"input_size={input_size}, layers={layers}, output_size={output_size} "
            f"batch_size={batch}, optimizer={optimizer_name}, lr={lr}, weight_decay={weight_decay}"
        )

        if ENABLE_PROGRESS_BARS and isinstance(pbar, tqdm):
            pbar.set_description(info_str)
            pbar.set_postfix(
                loss=total_loss,
                best_loss=best_loss,
                epochs_since_best=epochs_since_best,
            )

        logger.debug(info_str)
        logger.debug(
            f"Epoch: {epoch}, Loss: {total_loss}, Best Loss: {best_loss}, Epochs since best: {epochs_since_best}",
        )

    # reload the best performing model we found
    if best_state_dict is not None:
        logger.debug(f"Reloading best model from epoch {best_epoch}")
        model.load_state_dict(best_state_dict)
    else:
        logger.error("No best model found")

    # Pickle it as we'll forget the model architecture
    filename = f"kudos_models/kudos-{STUDY_VERSION}-{trial.number}.ckpt"
    with open(filename, "wb") as outfile:
        pickle.dump(model.to("cpu"), outfile)
        logger.debug(f"Saved model to {filename}")

    logger.debug(f"Finished trial {trial.number} with best_loss {best_loss}")
    return best_loss


def main() -> None:

    if args.test_model:
        low_predictions = test_one_by_one(args.test_model)
        if args.analyse:
            # Analyze with default 10% tolerance for floats
            results = analyze_dict_similarities(low_predictions)

            # Print results, showing only values that appear in at least 50% of dictionaries
            print_similarity_analysis(results, min_frequency=0.5)
        return

    if not ENABLE_TRAINING:
        return

    # Make our model output dir
    os.makedirs("kudos_models", exist_ok=True)

    # Create the database directory if it doesn't exist
    db_dir = os.path.dirname(os.path.abspath(args.db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    emmr_improvement_evaluator = EMMREvaluator()

    # By default, 1/100 of the median value of the 10th to 30th step of
    # emmr_improvement_evaluator
    median_error_evaluator = MedianErrorEvaluator(emmr_improvement_evaluator)

    # If the value of emmr_improvement_evaluator falls below the value of
    # median_error_evaluator, the early termination occurs.
    terminator = Terminator(
        improvement_evaluator=emmr_improvement_evaluator,
        error_evaluator=median_error_evaluator,
    )

    study = optuna.create_study(
        direction="minimize",
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=30, interval_steps=10),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        study_name=f"kudos_model_{STUDY_VERSION}",
        storage=DB_CONNECTION_STRING,
        load_if_exists=True,
        sampler=OPTUNA_SAMPLER,
    )
    try:
        study.optimize(
            objective,
            n_trials=NUMBER_OF_STUDY_TRIALS,
            callbacks=[TerminatorCallback(terminator)],
        )
    except (KeyboardInterrupt, AbortTrial):
        logger.warning("Trial process aborted")
    # fig = optuna.visualization.plot_terminator_improvement(
    #     study,
    #     plot_error=True,
    #     improvement_evaluator=emmr_improvement_evaluator,
    #     error_evaluator=median_error_evaluator,
    # )
    # fig.write_image(f"kudos_model_improvement_evaluator_{STUDY_VERSION}")
    # Print the best hyperparameters
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("Value: ", trial.value)
    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info(f"{key}: {value}")

    # Calculate the accuracy of the best model
    best_filename = f"kudos_models/kudos-{STUDY_VERSION}-{trial.number}.ckpt"
    # model = test_one_by_one(best_filename)
    logger.info(f"Best model file is: {best_filename}")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    random.seed()

    # Global constants now derived from args
    ENABLE_TRAINING = args.enable_training
    TRAINING_DATA_FILENAME = args.training_data
    VALIDATION_DATA_FILENAME = args.validation_data
    NUMBER_OF_STUDY_TRIALS = args.study_trials
    STUDY_VERSION = args.study_version
    ENABLE_PROGRESS_BARS = args.progress_bars

    if ENABLE_PROGRESS_BARS:
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        logger.add("logs/train.log", rotation="10 MB")

    # Create SQLite connection string
    DB_CONNECTION_STRING = f"sqlite:///{args.db_path}"
    main()
