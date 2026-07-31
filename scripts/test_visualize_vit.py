import csv
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import visualize_vit


class VisualizationDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def write_predictions(self, rows: list[list[object]]) -> Path:
        path = self.directory / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "index",
                    "true_label",
                    "predicted_label",
                    "confidence",
                    "prob_0",
                    "prob_1",
                ]
            )
            writer.writerows(rows)
        return path

    def test_reads_valid_predictions(self) -> None:
        path = self.write_predictions(
            [
                [0, 0, 0, 0.8, 0.8, 0.2],
                [1, 1, 1, 0.7, 0.3, 0.7],
            ]
        )

        predictions = visualize_vit.read_predictions(path)

        np.testing.assert_array_equal(predictions["index"], [0, 1])
        np.testing.assert_array_equal(predictions["predicted"], [0, 1])

    def test_rejects_non_contiguous_indices(self) -> None:
        path = self.write_predictions([[1, 0, 0, 0.8, 0.8, 0.2]])

        with self.assertRaisesRegex(ValueError, "contiguous"):
            visualize_vit.read_predictions(path)

    def test_reads_idx_images(self) -> None:
        path = self.directory / "images.idx3-ubyte"
        pixels = bytes(range(8))
        path.write_bytes(struct.pack(">IIII", 2051, 2, 2, 2) + pixels)

        images = visualize_vit.read_idx_images(path)

        self.assertEqual(images.shape, (2, 2, 2))
        np.testing.assert_array_equal(images.reshape(-1), np.arange(8))


if __name__ == "__main__":
    unittest.main()
