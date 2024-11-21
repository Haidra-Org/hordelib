import json
from os import PathLike
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


def load_dataset(path: PathLike | str) -> tuple[pd.DataFrame, np.ndarray]:
    with Path(path).open() as f:
        samples = json.load(f)

    data = {
        # Payload
        "sampler_name": [],
        "cfg_scale": [],
        "denoising_strength": [],
        "height": [],
        "width": [],
        "post_processing_0": [],
        "post_processing_1": [],
        "post_processing_2": [],
        "post_processing_order": [],
        "tiling": [],
        "hires_fix": [],
        "hires_fix_denoising_strength": [],
        "clip_skip": [],
        "control_type": [],
        "image_is_control": [],
        "return_control_map": [],
        "facefixer_strength": [],
        "lora_0": [],
        "lora_1": [],
        "lora_2": [],
        "lora_3": [],
        "lora_4": [],
        "ti_0": [],
        "ti_1": [],
        "ti_2": [],
        "ti_3": [],
        "ti_4": [],
        "workflow": [],
        "transparent": [],
        "use_nsfw_censor": [],
        "ddim_steps": [],
        "n_iter": [],
        "scheduler": [],
        "lora_count": [],
        "ti_count": [],
        # General
        "model": [],
        "source_processing": [],
        "model_baseline": [],
        "extra_source_images_count": [],
        "extra_source_images_combined_size": [],
        "source_image_size": [],
        "source_mask_size": [],
        # Request
        "state": [],
        "censored": [],
        "time_popped": [],
    }

    time_to_generate = []

    # Extract the name from the array
    def get_clean_arr(_payload: dict, _key: str):
        arr = _payload[_key]
        if not arr:
            return []
        return [x["name"] for x in arr]

    for sample in samples:
        sdk_api_job_info = sample["sdk_api_job_info"]
        payload = sdk_api_job_info["payload"]

        # Flatten post_processing
        post_processing = payload.get("post_processing", [])
        for i in range(3):
            payload[f"post_processing_{i}"] = (
                post_processing[i] if i < len(post_processing) else None
            )

        # Flatten loras
        loras = get_clean_arr(payload, "loras")
        for i in range(5):
            name = loras[i] if i < len(loras) else None
            payload[f"lora_{i}"] = name

        # Flatten tis
        tis = get_clean_arr(payload, "tis")
        for i in range(5):
            name = tis[i] if i < len(tis) else None
            payload[f"ti_{i}"] = name

        # Add to data
        for key in data:
            if key in payload:
                data[key].append(payload[key])
            elif key in sdk_api_job_info:
                data[key].append(sdk_api_job_info[key])
            elif key in sample:
                data[key].append(sample[key])
            else:
                data[key].append(None)

        # Add label
        time_to_generate.append(sample["time_to_generate"])

    df = pd.DataFrame(data)

    # Remove entries with unknown time
    labels = np.asarray(time_to_generate, dtype=np.float32)
    mask = ~np.isnan(labels) & (labels < 120)
    df: pd.DataFrame = df[mask]  # pyright: ignore [reportAssignmentType]
    labels = labels[mask]

    # Convert times to hour of the day
    df["time_popped"] = (df["time_popped"] % 86400 / 3600).astype(int)

    df["pixels"] = df["height"] * df["width"]
    df["pixels_steps"] = df["height"] * df["width"] * df["ddim_steps"]

    return df, labels


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    max_val = np.maximum(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    percentage_accuracy = (1 - diff / max_val) * 100
    return float(np.mean(percentage_accuracy)), float(np.mean(np.power(diff, 2)))


def main():
    train_dataset, train_labels = load_dataset("inference-time-data.json")
    val_dataset, val_labels = load_dataset("inference-time-data-validation.json")

    # Convert categorical string columns to integers
    categorical_cols = train_dataset.select_dtypes(include="object").columns
    lookup_maps = {}
    for col in categorical_cols:
        lookup_maps[col] = {
            value: index for index, value in enumerate(train_dataset[col].unique())
        }

    for col in categorical_cols:
        lookup_map = lookup_maps[col]
        train_dataset[col] = train_dataset[col].map(lookup_map).fillna(-1).astype(int)
        val_dataset[col] = val_dataset[col].map(lookup_map).fillna(-1).astype(int)

    # Print rough overview of the data
    for col in train_dataset.columns:
        print(f"{col}: {len(train_dataset[col].unique())} unique values")

    def train(
        params: dict, columns: list[str] | None = None
    ) -> tuple[lgb.Booster, np.ndarray]:
        if columns is None:
            columns = [
                "pixels_steps",
                "sampler_name",
                "cfg_scale",
                "post_processing_0",
                "hires_fix",
                "control_type",
                "lora_0",
                "transparent",
                "ddim_steps",
                "lora_count",
                "ti_count",
                "model",
                "model_baseline",
                "extra_source_images_count",
            ]

        # Wrap the data in a LightGBM Dataset
        train_data = lgb.Dataset(train_dataset[columns], label=train_labels)
        val_data = lgb.Dataset(val_dataset[columns], label=val_labels)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
        )

        y_pred = np.asarray(
            model.predict(val_dataset[columns], num_iteration=model.best_iteration)
        )

        print(model.best_iteration, y_pred.min(), y_pred.max())

        return model, y_pred

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": trial.suggest_categorical(
                "metric",
                ["l2", "l1", "rmse", "quantile"],
            ),
            "boosting_type": "gbdt",
            "early_stopping_round": 10,
            "device_type": "cpu",
            "verbosity": -1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
            "learning_rate": trial.suggest_float("learning_rate", 0.0, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 128),
        }

        _, y_pred = train(params)

        return evaluate(val_labels, y_pred)[1]

    def evaluate_best():
        # Best model found
        best_params = {
            "objective": "regression",
            "metric": "huber",
            "boosting_type": "dart",
            "verbosity": -1,
            "num_leaves": 256,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "min_child_samples": 20,
        }

        best_model, y_pred = train(best_params)

        accuracy, mse = evaluate(val_labels, y_pred)
        print(f"Accuracy: {accuracy}, MSE: {mse}")
        print("Best iteration:", best_model.best_iteration)

        lgb.plot_importance(
            best_model,
            figsize=(7, 6),
            title="Feature Importances",
        )
        plt.show()

    evaluate_best()

    # Optimize
    study = optuna.create_study(
        direction="minimize", storage="sqlite:///db.sqlite3"
    )
    study.optimize(objective, n_trials=100000, n_jobs=4, show_progress_bar=True)


if __name__ == "__main__":
    main()
