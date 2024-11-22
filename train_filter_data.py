import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class GenerationTimeStats:
    count: int
    times: list[float]
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    time_difference: float


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filters out outlier entries in kudos training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        default="inference-time-data.json",
        help="Path to JSON file. Must be single file",
    )

    parser.add_argument(
        "-m",
        "--max_deviation_seconds",
        type=float,
        default=5,
        help="The amount of deviation from the mean to allow before considering an entry an outlier and removing it.",
    )

    args = parser.parse_args()

    # Validate train ratio
    if args.max_deviation_seconds == 0:
        parser.error("max_deviation_seconds must be above 0")

    return args


def load_data(filename: str) -> list[dict[str, Any]]:
    """
    Loads the inference time data from a JSON file.
    """
    try:
        with open(filename) as f:
            data = json.load(f)

        # Handle both single dictionary and list of dictionaries
        if isinstance(data, dict):
            return [data]
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' contains invalid JSON.")
        exit(1)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        exit(1)


def create_param_key(entry: dict[str, Any]) -> str:
    """
    Creates a unique key for parameter comparison by including all relevant fields
    except time-related ones.
    """
    sdk_info = entry["sdk_api_job_info"]
    # Convert payload to sorted string to ensure consistent ordering
    payload_str = json.dumps(sdk_info["payload"], sort_keys=True)

    key_parts = [
        sdk_info["model"],
        sdk_info["source_processing"],
        sdk_info["model_baseline"],
        str(sdk_info["extra_source_images_count"]),
        str(sdk_info["extra_source_images_combined_size"]),
        str(sdk_info["source_image_size"]),
        str(sdk_info["source_mask_size"]),
        payload_str,
    ]

    return "||".join(key_parts)


def analyze_and_filter_entries(
    entries: list[dict[str, Any]],
    max_deviation_seconds: float = 5.0,
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """
    Analyzes entries and filters those that deviate too much from their group mean.
    Returns both filtered entries and removed outliers.
    """
    # Group entries by their parameters
    param_groups = defaultdict(list)
    for entry in entries:
        if entry["time_to_generate"] is None:
            continue
        if entry["state"] == "faulted":
            continue
        param_key = create_param_key(entry)
        param_groups[param_key].append((entry["time_to_generate"], entry))

    # Filter entries within each group
    filtered_entries: list = []
    outliers: defaultdict = defaultdict(dict)

    for param_key, times_and_entries in param_groups.items():
        if len(times_and_entries) <= 1:
            filtered_entries.extend(entry for _, entry in times_and_entries)
            continue

        times = [t for t, _ in times_and_entries]
        mean_time = mean(times)

        # Prepare group info
        group_info = {
            "total_entries": len(times_and_entries),
            "mean_time": mean_time,
            "outliers": [],
        }

        # Separate entries into kept and outliers
        for time, entry in times_and_entries:
            if abs(time - mean_time) <= max_deviation_seconds:
                filtered_entries.append(entry)
            else:
                group_info["outliers"].append((time, entry))

        # Only add to outliers if there are any outliers
        if group_info["outliers"]:
            outliers[param_key] = group_info

    return filtered_entries, outliers


def print_outlier_analysis(outliers: dict[str, dict]) -> None:
    """
    Prints analysis of the outlier entries.
    """
    if not outliers:
        print("No outliers found.")
        return

    print(f"\nFound outliers in {len(outliers)} parameter combinations:")

    for param_key, group_info in outliers.items():
        # Get model and processing type for this group
        model, source_processing = param_key.split("||")[:2]

        # Extract outlier times
        outlier_times = [t for t, _ in group_info["outliers"]]

        print("\nOutlier Group:")
        print(f"  Model: {model}")
        print(f"  Processing Type: {source_processing}")
        print(f"  Total entries in group: {group_info['total_entries']}")
        print(f"  Group mean time: {group_info['mean_time']:.3f}")
        print(f"  Number of outliers: {len(group_info['outliers'])}")
        print(f"  Outlier times: {[round(t, 3) for t in sorted(outlier_times)]}")
        print(f"  Outlier range: {min(outlier_times):.3f} - {max(outlier_times):.3f}")
        print(f"  Outliers removed: {round(len(group_info['outliers'])/group_info['total_entries']*100, 1)}%")


def save_filtered_data(entries: list[dict[str, Any]], input_filename: str) -> str:
    """
    Saves the filtered entries to a new JSON file.
    """
    input_path = Path(input_filename)
    output_filename = f"{input_path.stem}.analyzed{input_path.suffix}"

    with open(output_filename, "w") as f:
        json.dump(entries, f, indent=4)

    return output_filename


def print_summary(original_count: int, filtered_count: int, output_file: str) -> None:
    """
    Prints a summary of the filtering operation.
    """
    removed_count = original_count - filtered_count
    print("\nAnalysis Summary:")
    print(f"Original entries: {original_count}")
    print(f"Entries after filtering: {filtered_count}")
    print(f"Removed entries: {removed_count}")
    print(f"Removal percentage: {(removed_count/original_count)*100:.1f}%")
    print(f"\nFiltered data saved to: {output_file}")


def main():
    args = parse_args()
    filename = args.input_file
    entries = load_data(filename)
    original_count = len(entries)
    print(f"Loaded {original_count} entries from {filename}")

    # Filter entries and get outliers
    filtered_entries, outliers = analyze_and_filter_entries(entries, max_deviation_seconds=args.max_deviation_seconds)

    # Print outlier analysis
    print_outlier_analysis(outliers)

    # Save filtered entries
    output_file = save_filtered_data(filtered_entries, filename)

    # Print summary
    print_summary(original_count, len(filtered_entries), output_file)


if __name__ == "__main__":
    main()
