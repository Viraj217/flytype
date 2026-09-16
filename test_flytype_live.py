import base64
import unittest

import numpy as np

from flytype_live import serialize_event


class LiveTelemetryTests(unittest.TestCase):
    def test_spike_payload_uses_display_mapping_and_selected_counts(self):
        remap = np.array([-1, 0, 2, 1], dtype=np.int32)
        event = {
            "type": "inference",
            "prediction": "a",
            "spikes": np.array([9, 2, 0, 5], dtype=np.float32),
            "crop": np.zeros((69, 44), dtype=np.float32),
            "retina_rates": np.array([0, 90, 180], dtype=np.float32),
        }
        message = serialize_event(event, remap, np.array([1, 3]))
        indices = np.frombuffer(base64.b64decode(message["activity_indices"]), dtype=np.uint32)
        counts = np.frombuffer(base64.b64decode(message["activity_counts"]), dtype=np.uint8)
        retina = np.frombuffer(base64.b64decode(message["retina_rates"]), dtype=np.uint8)
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(counts, [2, 5])
        self.assertEqual(message["firing_neurons"], 3)
        self.assertEqual(message["total_spikes"], 16)
        self.assertEqual(message["selected_firing"], 2)
        self.assertEqual(message["selected_spikes"], 7)
        np.testing.assert_array_equal(retina, [0, 127, 255])
        self.assertEqual(message["retina_columns"], 3)
        self.assertEqual(message["retina_mean_hz"], 90.0)
        self.assertTrue(message["crop"])
        self.assertNotIn("spikes", message)


if __name__ == "__main__":
    unittest.main()
