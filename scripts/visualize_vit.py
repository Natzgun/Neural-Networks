#!/usr/bin/env python3
"""Generate ViT training and prediction visualizations from exported CSV files."""

import argparse
import csv
import struct
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create loss/accuracy curves, a confusion matrix, and a prediction "
            "grid from a trained C++ ViT."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "fashionmnist"),
        default="mnist",
        help="Dataset used during C++ training (default: mnist).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory containing metrics.csv and predictions.csv.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Optional path to the IDX test image file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of images in the prediction grid (default: 25).",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Show only incorrectly classified samples in the image grid.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used only to select images for the prediction grid (default: 42).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive plot windows in addition to saving PNG files.",
    )
    return parser.parse_args()


def default_images_path(dataset: str) -> Path:
    if dataset == "mnist":
        return Path("datasets/mnist/t10k-images.idx3-ubyte")
    return Path("datasets/fashionmnist/t10k-images-idx3-ubyte")


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Metrics file has no epochs: {path}")

    columns = (
        "epoch",
        "seed",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "time_ms",
    )
    return {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in columns
    }


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Predictions file is empty: {path}")

    required_columns = {"index", "true_label", "predicted_label", "confidence"}
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Predictions file is missing columns: {missing}")

    probability_columns = sorted(
        (name for name in fieldnames if name.startswith("prob_")),
        key=lambda name: int(name.removeprefix("prob_")),
    )
    expected_probability_columns = [
        f"prob_{index}" for index in range(len(probability_columns))
    ]
    if not probability_columns or probability_columns != expected_probability_columns:
        raise ValueError("Probability columns must be contiguous from prob_0.")

    probabilities = np.asarray(
        [[float(row[column]) for column in probability_columns] for row in rows],
        dtype=np.float32,
    )
    predictions = {
        "index": np.asarray([int(row["index"]) for row in rows], dtype=np.int64),
        "true": np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64),
        "predicted": np.asarray(
            [int(row["predicted_label"]) for row in rows], dtype=np.int64
        ),
        "confidence": np.asarray(
            [float(row["confidence"]) for row in rows], dtype=np.float32
        ),
        "probabilities": probabilities,
    }
    validate_predictions(predictions)
    return predictions


def validate_predictions(predictions: dict[str, np.ndarray]) -> None:
    indices = predictions["index"]
    true_labels = predictions["true"]
    predicted_labels = predictions["predicted"]
    confidence = predictions["confidence"]
    probabilities = predictions["probabilities"]
    classes = probabilities.shape[1]

    expected_indices = np.arange(indices.size, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("Prediction indices must be contiguous and start at zero.")
    if np.any((true_labels < 0) | (true_labels >= classes)):
        raise ValueError("True labels are outside the probability column range.")
    if np.any((predicted_labels < 0) | (predicted_labels >= classes)):
        raise ValueError("Predicted labels are outside the probability column range.")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(confidence)):
        raise ValueError("Predictions contain non-finite probabilities or confidence.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be between zero and one.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Probability rows must sum to one.")

    expected_labels = np.argmax(probabilities, axis=1)
    expected_confidence = probabilities[np.arange(indices.size), expected_labels]
    if not np.array_equal(predicted_labels, expected_labels):
        raise ValueError("Predicted labels do not match the probability argmax.")
    if not np.allclose(confidence, expected_confidence, atol=1e-5):
        raise ValueError("Confidence does not match the predicted probability.")


def read_idx_images(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"IDX image file not found: {path}")

    with path.open("rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid IDX image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Expected IDX image magic 2051, found {magic}")
        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    expected = count * rows * cols
    if pixels.size != expected:
        raise ValueError(
            f"IDX pixel count mismatch: expected {expected}, found {pixels.size}"
        )
    return pixels.reshape(count, rows, cols)


def plot_training_curves(metrics: dict[str, np.ndarray], output_path: Path) -> None:
    epochs = metrics["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, metrics["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, metrics["test_loss"], marker="o", label="Test")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, metrics["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, metrics["test_accuracy"], marker="o", label="Test")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(
        f"Vision Transformer training history (seed={int(metrics['seed'][0])})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")


def build_confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray, classes: int
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    axis.set(
        title="Confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", rotation=45)

    threshold = normalized.max() / 2.0 if normalized.size else 0.5
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                f"{matrix[row, col]}\n{normalized[row, col]:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalized[row, col] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")


def select_samples(
    predictions: dict[str, np.ndarray], count: int, errors_only: bool, seed: int
) -> np.ndarray:
    candidates = np.arange(predictions["index"].size)
    if errors_only:
        candidates = candidates[predictions["true"] != predictions["predicted"]]
        if candidates.size == 0:
            raise ValueError("There are no incorrect predictions to display.")

    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=min(count, candidates.size), replace=False)


def plot_prediction_grid(
    images: np.ndarray,
    predictions: dict[str, np.ndarray],
    selected: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    columns = min(5, selected.size)
    rows = int(np.ceil(selected.size / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 3.0 * rows))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, row_index in zip(axes_array, selected):
        image_index = predictions["index"][row_index]
        true_label = predictions["true"][row_index]
        predicted_label = predictions["predicted"][row_index]
        confidence = predictions["confidence"][row_index]
        correct = true_label == predicted_label

        axis.imshow(images[image_index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"Pred: {class_names[predicted_label]} ({confidence:.1%})\n"
            f"Real: {class_names[true_label]}",
            color="forestgreen" if correct else "firebrick",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_array[selected.size :]:
        axis.axis("off")

    figure.suptitle("ViT predictions: green = correct, red = incorrect")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir or Path("out/vit") / args.dataset
    images_path = args.images or default_images_path(args.dataset)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(artifacts_dir / "metrics.csv")
    predictions = read_predictions(artifacts_dir / "predictions.csv")
    images = read_idx_images(images_path)
    if predictions["index"].size != images.shape[0]:
        raise ValueError(
            "Predictions must contain exactly one row for every IDX test image."
        )

    class_names = [str(index) for index in range(10)]
    if args.dataset == "fashionmnist":
        class_names = FASHION_LABELS

    selected = select_samples(
        predictions, max(1, args.samples), args.errors_only, args.sample_seed
    )
    confusion = build_confusion_matrix(
        predictions["true"], predictions["predicted"], len(class_names)
    )

    curve_path = artifacts_dir / "training_curves.png"
    confusion_path = artifacts_dir / "confusion_matrix.png"
    grid_path = artifacts_dir / "prediction_grid.png"
    plot_training_curves(metrics, curve_path)
    plot_confusion_matrix(confusion, class_names, confusion_path)
    plot_prediction_grid(images, predictions, selected, class_names, grid_path)

    accuracy = np.mean(predictions["true"] == predictions["predicted"])
    print(f"Test accuracy from predictions: {accuracy:.4f}")
    print(f"Training curves: {curve_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Prediction grid: {grid_path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
