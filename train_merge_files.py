"""
train.py training json file merger

This script combines multiple kudos_training_data.json files provided by the AI Horde reGen worker
 containing lists of image generation dictionaries and splits them into training and validation datasets.

Features:
- Combines multiple JSON files into a unified dataset
- Performs random splitting with configurable ratios
- Ensures no overlap between training and validation sets
- Supports various input methods (directory, glob pattern, single file)
- Preserves JSON formatting in output files

Usage:
    python script.py INPUT_PATH [options]

Arguments:
    INPUT_PATH              Path to JSON files. Can be:
                           - A directory containing JSON files
                           - A glob pattern (e.g., "data/*.json")
                           - A single JSON file

Options:
    --train-ratio FLOAT    Ratio of data to use for training (default: 0.9)
    --output-train PATH    Path for training data output file
                          (default: inference-time-data.json)
    --output-val PATH      Path for validation data output file
                          (default: inference-time-data-validation.json)
    --seed INT            Random seed for reproducibility (default: 42)

Example:
    python script.py /path/to/json/files/ --train-ratio 0.8 --seed 123

Notes:
    - Input JSON files must contain lists of dictionaries
    - The script will automatically create output directories if they don't exist
    - Progress and error messages are printed to stdout
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path


def load_and_combine_json_files(input_path):
    """
    Load and combine all JSON files matching the pattern into a single list.
    """
    all_data = []

    # Convert input path to Path object and resolve any relative paths
    path = Path(input_path).resolve()

    # If path is a directory, look for all .json files
    if path.is_dir():
        json_files = list(path.glob("**/*.json"))
    # If path includes wildcards, use glob
    elif "*" in str(path):
        json_files = [Path(p) for p in glob.glob(str(path))]
    # If path is a single file
    elif path.is_file():
        json_files = [path]
    else:
        raise ValueError(f"Invalid path: {input_path}")

    if not json_files:
        raise ValueError(f"No JSON files found at: {input_path}")

    # Load each file and extend the main list
    for file_path in json_files:
        try:
            print(f"Processing: {file_path}")
            with open(file_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: Skipping {file_path} as it doesn't contain a list")
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

    return all_data


def split_and_save_data(
    data,
    train_ratio=0.9,
    train_file="inference-time-data.json",
    val_file="inference-time-data-validation.json",
):
    """
    Randomly split data into train and validation sets and save to files.
    """
    # Shuffle the data
    random.shuffle(data)

    # Calculate split index
    split_idx = int(len(data) * train_ratio)

    # Split the data
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create output directories if they don't exist
    for file_path in [train_file, val_file]:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    # Save training data
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)

    # Save validation data
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)

    return len(train_data), len(val_data)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Combine JSON files and split into training and validation sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_path", help="Path to JSON files. Can be a directory, a single file, or a glob pattern")

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (between 0 and 1)",
    )

    parser.add_argument(
        "--output-train",
        default="inference-time-data.json",
        help="Path for the training data output file",
    )

    parser.add_argument(
        "--output-val",
        default="inference-time-data-validation.json",
        help="Path for the validation data output file",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        parser.error("Train ratio must be between 0 and 1")

    return args


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    try:
        # Load and combine all JSON files
        print(f"\nLoading and combining JSON files from: {args.input_path}")
        combined_data = load_and_combine_json_files(args.input_path)

        if not combined_data:
            print("Error: No data was loaded!")
            return None

        print(f"\nTotal number of elements: {len(combined_data)}")

        # Split and save the data
        train_size, val_size = split_and_save_data(
            combined_data,
            train_ratio=args.train_ratio,
            train_file=args.output_train,
            val_file=args.output_val,
        )

        print("\nData split complete:")
        print(f"Training set size: {train_size} elements ({args.train_ratio*100:.1f}%)")
        print(f"Validation set size: {val_size} elements ({(1-args.train_ratio)*100:.1f}%)")
        print("\nFiles saved as:")
        print(f"Training data: {args.output_train}")
        print(f"Validation data: {args.output_val}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()
