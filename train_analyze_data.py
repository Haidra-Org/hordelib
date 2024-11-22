import json
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median, stdev
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
    time_difference: float  # Added field for time difference


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


def analyze_generation_times(
    entries: list[dict[str, Any]],
    min_time_difference: float = 1.0,
) -> dict[str, GenerationTimeStats]:
    """
    Analyzes the entries to find sets with identical parameters but different generation times.
    Only includes sets where the difference between min and max times exceeds min_time_difference.
    """
    # Group entries by their parameters
    param_groups = defaultdict(list)
    for entry in entries:
        if entry["time_to_generate"] is None:
            continue
        if entry["state"] == "faulted":
            continue
        param_key = create_param_key(entry)
        param_groups[param_key].append(entry["time_to_generate"])

    # Filter for groups with significant time differences
    different_times = {}
    for param_key, times in param_groups.items():
        if len(times) > 1:
            time_difference = max(times) - min(times)
            if time_difference > min_time_difference:
                different_times[param_key] = GenerationTimeStats(
                    count=len(times),
                    times=sorted(times),
                    mean_time=mean(times),
                    median_time=median(times),
                    std_dev=stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    time_difference=time_difference,
                )

    # Sort by time difference (largest first)
    return dict(sorted(different_times.items(), key=lambda x: x[1].time_difference, reverse=True))


def print_analysis(stats: dict[str, GenerationTimeStats]) -> None:
    """
    Prints a formatted analysis of the generation time statistics.
    """
    if not stats:
        print("No parameter combinations found with time differences greater than 1 second.")
        return

    print(f"Found {len(stats)} parameter combinations with significant time differences:\n")

    for i, (param_key, stat) in enumerate(stats.items(), 1):
        # Extract model name and source processing from the param_key for more readable output
        model, source_processing = param_key.split("||")[:2]

        print(f"Parameter Set {i}:")
        print(f"  Model: {model}")
        print(f"  Processing Type: {source_processing}")
        print(f"  Number of occurrences: {stat.count}")
        print(f"  Time difference (max-min): {stat.time_difference:.3f} seconds")
        print(f"  Generation times: {[round(t, 3) for t in stat.times]}")
        print(f"  Mean time: {stat.mean_time:.3f}")
        print(f"  Median time: {stat.median_time:.3f}")
        print(f"  Standard deviation: {stat.std_dev:.3f}")
        print(f"  Range: {stat.min_time:.3f} - {stat.max_time:.3f}")
        print()


def main():
    filename = "inference-time-data.json"
    entries = load_data(filename)

    print(f"Loaded {len(entries)} entries from {filename}")
    stats = analyze_generation_times(entries, min_time_difference=10.0)
    print_analysis(stats)


if __name__ == "__main__":
    main()
