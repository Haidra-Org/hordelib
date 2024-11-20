# Train a predictive model from horde payload inputs to predict inference time.
#
# Supports multi-processing, just run this multiple times and the processes will
# automatically work together on the training. We are training with torch and searching
# through network hyper parameters using Optuna.
#
# Requires two input files (both exactly the same format) which can be created by enabling
# the SAVE_KUDOS_TRAINING_DATA constant in the worker.
#   - inference-time-data.json
#   - inference-time-data-validation.json
#
# The output is a series of model checkpoints, "kudos_models/kudos-X-n.ckpt" Where n is the
# number of the trial and X is the study version. Once the best trial number is identified
# simply select the appropriate file.
#
# The stand-alone class in examples/kudos.py is the code to actually use the model.
#
# Requires also a local mysql database named "optuna" and assumes it can connect
# with user "root" password "root". Change to your needs.
#
# For visualisation with optuna dashboard:
#   pip install optuna-dashboard
#
#   optuna-dashboard sqlite:///optuna_studies.db
#   or
#   optuna-dashboard mysql://root:root@localhost/optuna
#
# This is a quick hack to assist with kudos calculation.
import argparse
import json
import math
import os
import pickle
import random
import signal
import sys
import time
from collections import defaultdict
from typing import Any

import optuna
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from optuna.terminator import EMMREvaluator, MedianErrorEvaluator, Terminator, TerminatorCallback
from torch import optim
from torch.utils.data import DataLoader, Dataset


from hordelib.horde import HordeLib

random.seed()

# Number of trials to run.
# Each trial generates a new neural network topology with new hyper parameters and trains it.
NUMBER_OF_STUDY_TRIALS = 300

# Hyper parameter search bounds
NUM_EPOCHS = 2000
PATIENCE = 100  # if no improvement in this many epochs, stop early
MIN_NUMBER_OF_EPOCHS = 50
MIN_HIDDEN_LAYERS = 1
MAX_HIDDEN_LAYERS = 9
MIN_NODES_IN_LAYER = 4
MAX_NODES_IN_LAYER = 128
MIN_LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 1e-2
MIN_WEIGHT_DECAY = 1e-6
MAX_WEIGHT_DECAY = 1e-1
MIN_DATA_BATCH_SIZE = 32
MAX_DATA_BATCH_SIZE = 256

DB_PATH = "optuna_studies.db"

# The study sampler to use
OPTUNA_SAMPLER = optuna.samplers.TPESampler(n_startup_trials=12, n_ei_candidates=20)
# # OPTUNA_SAMPLER = optunahub.load_module("samplers/auto_sampler").AutoSampler()
# HEBOSampler = optunahub.load_module("samplers/hebo").HEBOSampler
# OPTUNA_SAMPLER = HEBOSampler(
#     {
#         "x": optuna.distributions.FloatDistribution(-10, 10),
#         "y": optuna.distributions.IntDistribution(-10, 10),
#     },
# )
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
        default="./inference-time-data.json",
        help="Path to training data file",
    )

    parser.add_argument(
        "--validation-data",
        type=str,
        default="./inference-time-data-validation.json",
        help="Path to validation data file",
    )

    # Study parameters
    parser.add_argument("--study-trials", type=int, default=2000, help="Number of trials to run")

    parser.add_argument("-v", "--study-version", type=str, default="v25", help="Version number of the study")

    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of workers to use")

    parser.add_argument("--notebook", action="store_true", default=False, help="Run in notebook mode")

    return parser.parse_args()


class AbortTrial(Exception):
    pass


def signal_handler(sig, frame):
    raise AbortTrial


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# This is an example of how to use the final model, pass in a horde payload, get back a predicted time in seconds
def payload_to_time(model, payload) -> float:
    """Convert a horde payload to a time prediction using the model.

    Args:
        model (nn.Module): The PyTorch model
        payload (dict): The horde payload

    Returns:
        float: The predicted time in seconds
    """
    inputs = KudosDataset.payload_to_tensor(payload).squeeze()
    with torch.no_grad():
        output = model(inputs)
    return round(float(output.item()), 2)


# This is how to load the model required above
def load_model(model_filename):
    """Load a model from a file."""
    print(f"Loading model from {model_filename}")
    with open(model_filename, "rb") as infile:
        return pickle.load(infile)


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

    items = []
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

        value_groups = defaultdict(int)
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


def print_similarity_analysis(
    results: dict[str, list[tuple[Any, float]]],
    min_frequency: float = 0.5,
) -> None:
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


def test_one_by_one(
    model_filename: str,
) -> list[dict]:
    """Test the model against the validation dataset and return a list of bad predictions.

    Also prints out some statistics about the model's performance.

    Args:
        model_filename (str): The json file containing the model

    Returns:
        list[dict]: A list of bad predictions
    """

    dataset = []
    with open(VALIDATION_DATA_FILENAME) as infile:
        d = json.load(infile)
        for p in d:
            if p["time_to_generate"] is not None:
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
    """A PyTorch dataset for the Kudos training data.

    Use payload_to_tensor to convert a horde payload to a tensor for training.
    """

    def __init__(self, filename):
        """Initialise the dataset.

        Payloads with time_to_generate > 180 seconds are skipped.

        Args:
            filename (str): The filename of the training data

        Returns:
            KudosDataset: The dataset object
        """
        self.data = []
        self.labels = []

        with open(filename) as infile:
            payload_list = json.load(infile)

            skipped_payloads = 0
            for payload in payload_list:
                time_to_generate = payload["time_to_generate"]
                if time_to_generate is None:
                    continue

                if time_to_generate > 180:
                    skipped_payloads += 1
                    continue

                self.data.append(KudosDataset.payload_to_tensor(payload)[0])
                self.labels.append(time_to_generate)

            print(f"Skipped {skipped_payloads} payloads with time_to_generate > 180 seconds")

        self.labels = torch.tensor(self.labels).float()
        self.mixed_data = torch.stack(self.data)

    @classmethod
    def payload_to_tensor(cls, payload: dict) -> torch.Tensor:
        """Convert a horde payload to a tensor for training.

        Args:
            payload (dict): The horde payload

        Returns:
            torch.Tensor: The resulting tensor, of shape (1, n)
        """
        payload = payload["sdk_api_job_info"]
        p: dict = payload["payload"]
        data = []
        data_samplers = []
        data_control_types = []
        data_source_processing_types = []
        data_model_baseline = []
        data_post_processors = []
        data_schedulers = []
        data_workflows = []
        total_pixels = p["height"] * p["width"]
        hires_fix = p.get("hires_fix", False)
        if hires_fix:
            hires_fix_denoising_strength = p.get("hires_fix_denoising_strength", 0.65) or 0.65

        control_type = p.get("control_type", "None")
        has_control = control_type != "None"

        image_is_control = p.get("image_is_control", False)
        return_control_map = p.get("return_control_map", False)
        if not has_control:
            image_is_control = False
            return_control_map = False

        data.append(
            [
                total_pixels / (1024 * 1024),
                p["ddim_steps"] / 100,
                p["cfg_scale"] / 30,
                p.get("denoising_strength", 1.0) if p.get("denoising_strength", 1.0) is not None else 1.0,
                # p.get("clip_skip", 1.0) / 4,
                p.get("control_strength", 1.0) if p.get("control_strength", 1.0) is not None else 1.0,
                p.get("facefixer_strength", 1.0) if p.get("facefixer_strength", 1.0) is not None else 1.0,
                p.get("lora_count", 0.0) / 5,
                p.get("ti_count", 0.0) / 10,
                # p.get("extra_source_images_count", 0.0) / 5,
                # p.get("extra_source_images_combined_size", 0.0) / 100_000,
                # p.get("source_image_size", 0.0) / 100_000,
                # p.get("source_mask_size", 0.0) / 100_000,
                1.0 if hires_fix else 0.0,
                hires_fix_denoising_strength if hires_fix else 0.0,
                1.0 if has_control and image_is_control else 0.0,
                1.0 if has_control and return_control_map else 0.0,
                1.0 if p.get("transparent", True) else 0.0,
                1.0 if p.get("tiling", True) else 0.0,
                # 1.0 if p.get("post_processing_order", "facefixers_first") == "facefixers_first" else 0.0,
            ],
        )
        data_model_baseline.append(
            payload["model_baseline"] if payload["model_baseline"] in KNOWN_MODEL_BASELINES else "stable_diffusion_xl",
        )
        data_schedulers.append(p["scheduler"])
        data_samplers.append(p["sampler_name"] if p["sampler_name"] in KNOWN_SAMPLERS else "k_euler")
        data_control_types.append(
            control_type if control_type is not None else "None",
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
    """Create a PyTorch nn.Sequential model with the given layer sizes.

    Args:
        trial (optuna.Trial): The trial object
        layer_sizes (list[int]): The sizes of the hidden layers
        input_size (int): The size of the input layer
        output_size (int, optional): The size of the output layer. Defaults to 1.

    Returns:
        nn.Sequential: The PyTorch model with the given layer sizes
    """

    # Define the layer sizes
    layer_sizes = [input_size] + layer_sizes + [output_size]

    # Create the layers and activation functions
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())  # Use ReLU activation for all layers except the last one
            # Add a dropout layer
            if i > 0:
                drop = trial.suggest_float(f"dropout_l{i}", 0.05, 0.2, log=True)
                layers.append(nn.Dropout(drop))

    # Create the nn.Sequential model
    return nn.Sequential(*layers)


def build_dataloaders(
    train_dataset: KudosDataset,
    validate_dataset: KudosDataset,
    batch_sizes: list[int],
):
    """Build dataloaders for the specified batch sizes."""
    dataloaders = {}
    for batch_size in batch_sizes:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
        validate_loader = DataLoader(
            validate_dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
        )
        dataloaders[batch_size] = (train_loader, validate_loader)
    return dataloaders


def objective(
    trial: optuna.Trial,
    dataloaders: dict[int, tuple[DataLoader, DataLoader]],
) -> float:
    """Calculate the objective function for the trial.

    Args:
        trial (optuna.Trial): The trial object
        dataloaders (dict[int, tuple[DataLoader, DataLoader]]): Pre-built dataloaders

    Returns:
        float: The loss value from the best epoch
    """

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
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
    lr = trial.suggest_float("learning_rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE, log=True)
    weight_decay = trial.suggest_float("weight_decay", MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, log=True)

    optimizer = None

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer is None:
        raise Exception("Unknown optimizer")

    batch = trial.suggest_categorical("batch_size", list(dataloaders.keys()))
    train_loader, validate_loader = dataloaders[batch]

    # Loss function
    criterion = nn.L1Loss()

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler()

    total_loss = None
    best_epoch = best_loss = best_state_dict = None
    epochs_since_best = 0

    if NOTEBOOK_MODE:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")
    for epoch in pbar:
        # Train the model
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            with amp.autocast():
                outputs = model(data)
                loss: torch.Tensor = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            data: torch.Tensor
            labels: torch.Tensor
            for data, labels in validate_loader:
                data = data.to(device)
                labels = labels.to(device)
                with amp.autocast():
                    outputs = model(data)
                    labels = labels.unsqueeze(1)
                    loss = criterion(outputs, labels)
                    total_loss += loss

        total_loss /= len(validate_loader)
        total_loss = round(float(total_loss), 2)
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
            epochs_since_best = 0
        else:
            epochs_since_best = epoch - best_epoch
            if epochs_since_best >= PATIENCE:
                print(f"Early stopping at epoch {epoch} due to no improvement")
                break

        info_string = (
            f"layers={layers}, batch_size={batch}, optimizer={optimizer_name}, lr={lr}, "
            f"weight_decay={weight_decay} "
            f"input_size={input_size}, output_size={output_size}"
        )
        pbar.set_description(info_string)

        pbar.set_postfix(
            loss=total_loss,
            best_loss=best_loss,
            epochs_since_best=epochs_since_best,
        )

        pbar.update()

    print(f"Best loss: {best_loss} at epoch {best_epoch}. Using the model as of epoch {best_epoch}")
    model.load_state_dict(best_state_dict)

    # Pickle it as we'll forget the model architecture
    filename = f"kudos_models/kudos-{STUDY_VERSION}-{trial.number}.ckpt"
    with open(filename, "wb") as outfile:
        pickle.dump(model.to("cpu"), outfile)

    test_one_by_one(filename)

    return best_loss


def main(
    test_model: str | None = None,
    analyse: bool = False,
) -> None:
    random.seed()

    if test_model:
        low_predictions = test_one_by_one(test_model)
        if analyse:
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
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
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
    train_dataset = KudosDataset(TRAINING_DATA_FILENAME)
    train_dataset.mixed_data.to("cuda")
    train_dataset.labels.to("cuda")
    validate_dataset = KudosDataset(VALIDATION_DATA_FILENAME)
    validate_dataset.mixed_data.to("cuda")
    validate_dataset.labels.to("cuda")

    # Build dataloaders in advance
    batch_start = int(math.ceil(math.log2(MIN_DATA_BATCH_SIZE)))
    batch_end = int(math.floor(math.log2(MAX_DATA_BATCH_SIZE)))
    batch_sizes = [2**i for i in range(batch_start, batch_end + 1)]
    dataloaders = build_dataloaders(train_dataset, validate_dataset, batch_sizes)

    try:
        study.optimize(
            lambda trial: objective(trial, dataloaders),
            n_trials=NUMBER_OF_STUDY_TRIALS,
            callbacks=[TerminatorCallback(terminator)],
        )
    except (KeyboardInterrupt, AbortTrial):
        print("Aborting trial")
    except Exception as e:
        print(f"Exception: {e}")
    # fig = optuna.visualization.plot_terminator_improvement(
    #     study,
    #     plot_error=True,
    #     improvement_evaluator=emmr_improvement_evaluator,
    #     error_evaluator=median_error_evaluator,
    # )
    # fig.write_image(f"kudos_model_improvement_evaluator_{STUDY_VERSION}")
    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    # Calculate the accuracy of the best model
    best_filename = f"kudos_models/kudos-{STUDY_VERSION}-{trial.number}.ckpt"
    # model = test_one_by_one(best_filename)
    print(f"Best model file is: {best_filename}")


def set_globals(
    enable_training: bool,
    training_data_filename: str,
    validation_data_filename: str,
    number_of_study_trials: int,
    study_version: str,
    db_path: str,
    workers: int,
    notebook_mode: bool,
    test_model: bool = False,
):
    global ENABLE_TRAINING
    global TRAINING_DATA_FILENAME
    global VALIDATION_DATA_FILENAME
    global NUMBER_OF_STUDY_TRIALS
    global STUDY_VERSION
    global DB_PATH
    global DB_CONNECTION_STRING
    global NUM_WORKERS
    global NOTEBOOK_MODE
    global TEST_MODEL

    ENABLE_TRAINING = enable_training
    TRAINING_DATA_FILENAME = training_data_filename
    VALIDATION_DATA_FILENAME = validation_data_filename
    NUMBER_OF_STUDY_TRIALS = number_of_study_trials
    STUDY_VERSION = study_version
    DB_PATH = db_path
    DB_CONNECTION_STRING = f"sqlite:///{db_path}"
    NUM_WORKERS = workers
    NOTEBOOK_MODE = notebook_mode
    TEST_MODEL = test_model


if __name__ == "__main__":

    # Parse command line arguments
    args = parse_args()

    # Set random seed

    # Global constants now derived from args
    set_globals(
        args.enable_training,
        args.training_data,
        args.validation_data,
        args.study_trials,
        args.study_version,
        args.db_path,
        args.workers,
        args.notebook,
        args.test_model,
    )
    main(test_model=args.test_model, analyse=args.analyse)
