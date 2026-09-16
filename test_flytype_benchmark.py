import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from flytype_benchmark import checkpoint_counts, digest_image, raw_features, read_samples, select_features
from flytype_vision import pad_to_training_height


class BenchmarkTests(unittest.TestCase):
    def test_vertical_padding_preserves_glyph_and_is_idempotent(self):
        arr = np.full((59, 44), 52 / 255, dtype=np.float32)
        arr[24:42, 14:30] = 102 / 255
        padded = pad_to_training_height(arr, 69)
        self.assertEqual(padded.shape, (69, 44))
        np.testing.assert_array_equal(padded[5:64], arr)
        np.testing.assert_array_equal(padded[:5], np.full((5, 44), arr[0, 0]))
        self.assertIs(pad_to_training_height(padded, 69), padded)
        with self.assertRaises(ValueError):
            pad_to_training_height(np.zeros((70, 44)), 69)

    def test_short_row_keeps_training_baseline(self):
        training = np.full((69, 44), 52 / 255, dtype=np.float32)
        training[29:47, 14:30] = 102 / 255
        short_row = training[6:-5]
        self.assertEqual(short_row.shape, (58, 44))
        np.testing.assert_array_equal(pad_to_training_height(short_row, 69), training)

    def test_production_predict_normalizes_before_brain(self):
        from unittest.mock import Mock
        from flytype import FlyType
        fly = FlyType.__new__(FlyType)
        fly.run_brain = Mock(return_value=np.array([1, 2], dtype=np.float32))
        fly.top_idx = np.array([1])
        fly.model = Mock()
        fly.model.predict.return_value = np.array(["e"])
        self.assertEqual(fly.predict(np.zeros((59, 44), dtype=np.float32)), "e")
        self.assertEqual(fly.run_brain.call_args.args[0].shape, (69, 44))

    def test_infer_exposes_same_activity_used_by_classifier(self):
        from flytype import FlyType

        class Classifier:
            classes_ = np.array(["a", "b", "c"])

            def predict(self, features):
                np.testing.assert_array_equal(features, [[4, 2]])
                return np.array(["b"])

            def predict_proba(self, features):
                return np.array([[0.1, 0.8, 0.1]])

        fly = FlyType.__new__(FlyType)
        fly.run_brain = lambda arr: np.array([0, 4, 2], dtype=np.float32)
        fly.top_idx = np.array([1, 2])
        fly.model = Classifier()
        result = fly.infer(np.zeros((59, 44), dtype=np.float32))
        self.assertEqual(result["prediction"], "b")
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertEqual(result["alternatives"][0]["letter"], "b")
        np.testing.assert_array_equal(result["spikes"], [0, 4, 2])
        self.assertEqual(result["crop"].shape, (69, 44))

    def test_visual_cache_stability_compares_pixels_and_boundaries(self):
        from flytype import FlyType
        fly = FlyType.__new__(FlyType)
        glyph = np.zeros((59, 44), dtype=np.float32)
        cache = [("char", glyph), ("space", None)]
        self.assertTrue(fly.visual_cache_matches(cache, [("char", glyph.copy()), ("space", None)]))
        changed = glyph.copy()
        changed[0, 0] = 1
        self.assertFalse(fly.visual_cache_matches(cache, [("char", changed), ("space", None)]))
        self.assertFalse(fly.visual_cache_matches(cache, [("char", glyph)]))

    def test_checkpoint_matches_independent_simulator_runs(self):
        import scipy.sparse as sp
        from flysim import FlyBrain
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.npz"
            graph = sp.csr_matrix(np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0]], dtype=np.float32))
            annotation = np.array(["L1", "relay", "output"])
            np.savez(path, data=graph.data, indices=graph.indices, indptr=graph.indptr,
                     shape=graph.shape, bodies=np.arange(3), types=annotation,
                     superclass=annotation, subclass=annotation, receptor=annotation,
                     fru=annotation, nt=annotation)
            brain = FlyBrain(path)
            drive = {(0,): np.array([180], dtype=np.float32)}
            log = brain.run(drive, 200, seed=42, spike_log=True)["_spikes"]
            checkpoints = checkpoint_counts(log, brain.n, (50, 100, 200))
            self.assertGreater(int(checkpoints[200].sum()), 0)
            for steps in (50, 100, 200):
                independent = brain.run(drive, steps, seed=42, spike_log=True)["_spikes"]
                np.testing.assert_array_equal(checkpoints[steps], checkpoint_counts(independent, brain.n, (steps,))[steps])

    def test_raw_padding_preserves_pixels_and_rejects_clipping(self):
        image = Image.fromarray(np.array([[10, 90], [10, 10]], dtype=np.uint8))
        result = raw_features(image, (3, 2)).reshape(2, 3)
        np.testing.assert_allclose(result[:, :2], np.asarray(image) / 255)
        with self.assertRaises(ValueError):
            raw_features(image, (1, 2))

    def test_duplicate_hash_ignores_png_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a").mkdir()
            image = Image.new("L", (4, 4), 45)
            image.save(root / "a/1.png", compress_level=0)
            image.save(root / "a/2.png", compress_level=9)
            samples = read_samples(root)
            self.assertEqual(samples[0]["digest"], samples[1]["digest"])
            self.assertEqual(samples[0]["digest"], digest_image(image))

    def test_selection_retains_perfect_discriminator(self):
        import scipy.sparse as sp
        X = sp.csr_matrix(np.array([[0, 1, 3], [0, 2, 3], [1, 1, 3], [1, 2, 3]], dtype=float))
        selected = select_features(X, np.array(["a", "a", "b", "b"]), 1)
        np.testing.assert_array_equal(selected, [0])


if __name__ == "__main__":
    unittest.main()
