# type: ignore
#
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
#   optuna-dashboard mysql://root:root@localhost/optuna
#
# This is a quick hack to assist with kudos calculation.
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import hordelib

hordelib.initialise()
import pickle
from collections import defaultdict
from typing import Any

import optuna
import optunahub
import yaml
from optuna.terminator import EMMREvaluator, MedianErrorEvaluator, Terminator, TerminatorCallback
from tqdm import tqdm

from hordelib.horde import HordeLib


# We have the following inputs to our kudos calculation.
# The payload below is pruned from unused fields during tensor conversion
PAYLOAD_EXAMPLE = {
    "sdk_api_job_info": {
        "id_": "7ba3b75b-6926-4e78-ad42-6763fa15c262",
        "ids": ["7ba3b75b-6926-4e78-ad42-6763fa15c262"],
        "payload": {
            "sampler_name": "k_euler",
            "cfg_scale": 24.0,
            "denoising_strength": None,
            "seed": "2066405361",
            "height": 1024,
            "width": 768,
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
            "ddim_steps": 40,
            "n_iter": 1,
            "scheduler": "karras",
            "lora_count": 0,
            "ti_count": 0,
        },
        "model": "Dreamshaper",
        "source_processing": "img2img",
        "model_baseline": "stable_diffusion_1",
        "extra_source_images_count": 0,
        "extra_source_images_combined_size": 0,
        "source_image_size": 0,
        "source_mask_size": 0,
    },
    "state": "ok",
    "censored": False,
    "time_popped": 1729837827.8703332,
    "time_submitted": 1729837835.3562803,
    "time_to_generate": 4.450331687927246,
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


def load_config(config_file):
    with open(config_file) as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="ML Training Script with configurable parameters")

    # Configuration file
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML configuration file")

    # Training control
    parser.add_argument("-e", "--enable-training", action="store_true", default=False, help="Enable training mode")
    parser.add_argument(
        "-a",
        "--analyse",
        action="store_true",
        default=False,
        help="When true will analyse and report which values in the bad predictions are the most common",
    )

    # Test mode
    parser.add_argument("-t", "--test-model", type=str, help="Path to model file for testing one by one")

    return parser.parse_args()


# This is an example of how to use the final model, pass in a horde payload, get back a predicted time in seconds
def payload_to_time(model, payload):
    inputs = KudosDataset.payload_to_tensor(payload).squeeze()
    with torch.no_grad():
        output = model(inputs)
    return round(float(output.item()), 2)


# This is how to load the model required above
def load_model(model_filename):
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
            if p["time_to_generate"] is not None:
                dataset.append(p)

    model = load_model(model_filename)

    perc = []
    total_job_time = 0
    total_time = 0
    bad_predictions = []
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
        # Print the data if very inaccurate prediction
        if percentage_accuracy < 60:
            # print(data)
            bad_predictions.append(data)
        # print(f"{predicted} predicted, {actual} actual ({round(percentage_accuracy, 1)}%)")
    avg_perc = round(sum(perc) / len(perc), 1)
    print(f"Average kudos calculation time {round((total_time*1000000)/len(perc))} micro-seconds")
    print(f"Average actual job time in the dataset {round(total_job_time/len(perc), 2)} seconds")
    print(f"Average accuracy = {avg_perc}%")
    return bad_predictions


class KudosDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        self.labels = []

        with open(filename) as infile:
            payload_list = json.load(infile)

            for payload in payload_list:
                if payload["time_to_generate"] is None:
                    continue
                self.data.append(KudosDataset.payload_to_tensor(payload)[0])
                self.labels.append(payload["time_to_generate"])

        self.labels = torch.tensor(self.labels).float()
        self.mixed_data = torch.stack(self.data)

    @classmethod
    def payload_to_tensor(cls, payload):
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
                p["height"] / 1024,
                p["width"] / 1024,
                p["ddim_steps"] / 100,
                p["cfg_scale"] / 30,
                p.get("denoising_strength", 1.0) if p.get("denoising_strength", 1.0) is not None else 1.0,
                # p.get("clip_skip", 1.0) / 4,
                p.get("control_strength", 1.0) if p.get("control_strength", 1.0) is not None else 1.0,
                p.get("facefixer_strength", 1.0) if p.get("facefixer_strength", 1.0) is not None else 1.0,
                p.get("lora_count", 0.0) / 5,
                p.get("ti_count", 0.0) / 10,
                p.get("extra_source_images_count", 0.0) / 5,
                p.get("extra_source_images_combined_size", 0.0) / 100_000,
                p.get("source_image_size", 0.0) / 100_000,
                p.get("source_mask_size", 0.0) / 100_000,
                1.0 if p.get("hires_fix", True) else 0.0,
                p.get("hires_fix_denoising_strength", 0.65) or 0.0,
                1.0 if p.get("image_is_control", True) else 0.0,
                # 1.0 if p.get("return_control_map", True) else 0.0,
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


def create_sequential_model(trial, layer_sizes, input_size, output_size=1):
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
                drop = trial.suggest_float(f"dropout_l{i}", 0.05, 0.5, log=True)
                layers.append(nn.Dropout(drop))

    # Create the nn.Sequential model
    return nn.Sequential(*layers)


def objective(trial: optuna.Trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial.set_user_attr("name", "predict_kudos")

    # Network topology
    input_size = len(KudosDataset.payload_to_tensor(PAYLOAD_EXAMPLE)[0])
    num_hidden_layers = trial.suggest_int("hidden_layers", 1, MAX_HIDDEN_LAYERS, log=True)
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
    lr = trial.suggest_float("lr", MIN_LEARNING_RATE, MAX_LEARNING_RATE, log=True)
    weight_decay = trial.suggest_float("weight_decay", MIN_WEIGHT_DECAY, MAX_WEIGHT_DECAY, log=True)

    optimizer = None

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
    batch_start = int(math.ceil(math.log2(MIN_DATA_BATCH_SIZE)))
    batch_end = int(math.floor(math.log2(MAX_DATA_BATCH_SIZE)))
    batch_sizes = [2**i for i in range(batch_start, batch_end + 1)]
    batch = trial.suggest_categorical("batch_size", batch_sizes)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # Load the validation dataset
    validate_dataset = KudosDataset(VALIDATION_DATA_FILENAME)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=True)

    # Loss function
    criterion = nn.L1Loss()

    total_loss = None
    best_epoch = None
    best_loss = None
    best_state_dict = None

    # Print all of the hyperparameters of the trial
    trial_details_string = (
        f"Trial {trial.number} - Hidden layers: {num_hidden_layers}, Layer sizes: {layers}, "
        f"Optimizer: {optimizer_name}, Learning rate: {lr}, Weight decay: {weight_decay}, Batch size: {batch}"
    )
    optuna.logging.get_logger("optuna").info(trial_details_string)

    # Initialize tqdm progress bar
    pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")

    for epoch in pbar:
        # Train the model
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, labels in validate_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                total_loss += loss

        total_loss /= len(validate_loader)
        total_loss = round(float(total_loss), 2)

        epochs_since_best = 0

        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()

        else:
            epochs_since_best = epoch - best_epoch
            pbar.set_postfix(epochs_since_best=epochs_since_best)
            if epochs_since_best >= PATIENCE:
                # Stop early, no improvement in awhile
                break

        pbar.set_postfix(loss=total_loss, best_loss=best_loss, epochs_since_best=epochs_since_best)

    # reload the best performing model we found
    model.load_state_dict(best_state_dict)

    # Pickle it as we'll forget the model architecture
    filename = f"kudos_models/kudos-{STUDY_VERSION}-{trial.number}.ckpt"
    with open(filename, "wb") as outfile:
        pickle.dump(model.to("cpu"), outfile)

    optuna.logging.get_logger("optuna").info(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Best loss: {best_loss}")

    return best_loss


def main():
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

    if ENABLE_TRAINING:
        # Create the database directory if it doesn't exist
        db_dir = os.path.dirname(os.path.abspath(config["db_path"]))
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
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=30, interval_steps=10),
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
        except KeyboardInterrupt:
            print("Trial process aborted")
        fig = optuna.visualization.plot_terminator_improvement(
            study,
            plot_error=True,
            improvement_evaluator=emmr_improvement_evaluator,
            error_evaluator=median_error_evaluator,
        )
        fig.write_image(f"kudos_model_improvement_evaluator_{STUDY_VERSION}")
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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Set random seed
    random.seed()

    # The study sampler to use
    # OPTUNA_SAMPLER = optuna.samplers.TPESampler(n_startup_trials=30, n_ei_candidates=30)
    OPTUNA_SAMPLER = optunahub.load_module("samplers/auto_sampler").AutoSampler()
    # OPTUNA_SAMPLER = optuna.samplers.NSGAIISampler()  # genetic algorithm

    ENABLE_TRAINING = args.enable_training
    TRAINING_DATA_FILENAME = config["training_data"]
    VALIDATION_DATA_FILENAME = config["validation_data"]
    NUMBER_OF_STUDY_TRIALS = config["study_trials"]
    STUDY_VERSION = config["study_version"]
    DB_CONNECTION_STRING = f"sqlite:///{config['db_path']}"

    # Hyper parameter search bounds
    NUM_EPOCHS = int(config["num_epochs"])
    PATIENCE = int(config["patience"])
    MIN_NUMBER_OF_EPOCHS = int(config["min_number_of_epochs"])
    MAX_HIDDEN_LAYERS = int(config["max_hidden_layers"])
    MIN_NODES_IN_LAYER = int(config["min_nodes_in_layer"])
    MAX_NODES_IN_LAYER = int(config["max_nodes_in_layer"])
    MIN_LEARNING_RATE = float(config["min_learning_rate"])
    MAX_LEARNING_RATE = float(config["max_learning_rate"])
    MIN_WEIGHT_DECAY = float(config["min_weight_decay"])
    MAX_WEIGHT_DECAY = float(config["max_weight_decay"])
    MIN_DATA_BATCH_SIZE = int(config["min_data_batch_size"])
    MAX_DATA_BATCH_SIZE = int(config["max_data_batch_size"])
    main()
