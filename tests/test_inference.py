import unittest
from unittest.mock import patch

import numpy as np

from pendulum.inference import infer
from pendulum.model import PendulumModel


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = PendulumModel(hidden_size=8, num_hidden_layers=2)
        self.weights = self.model.state_dict()
        self.enterContext(patch("pendulum.inference.torch.cuda.is_available", return_value=False))
        self.enterContext(patch("pendulum.inference.PendulumModel", return_value=self.model))
        self.loader = self.enterContext(patch("pendulum.inference.torch.load", return_value=self.weights))
        self.enterContext(patch("pendulum.inference.tqdm", side_effect=lambda steps: steps))

    def test_timing_excludes_checkpoint_loading_and_warmup(self):
        clock = [0.0]
        original_forward = self.model.forward

        def load(*args, **kwargs):
            clock[0] += 1000.0
            return self.weights

        def forward(inputs):
            clock[0] += 1.0
            return original_forward(inputs)

        self.loader.side_effect = load
        with (
            patch.object(self.model, "forward", side_effect=forward) as predict,
            patch("pendulum.inference.perf_counter", side_effect=lambda: clock[0])
        ):
            theta, omega, elapsed = infer("unused.pth", num_steps=3, return_timing=True)

        self.assertEqual(elapsed, 3.0)
        self.assertEqual(predict.call_count, 4)  # One discarded warm-up and three predictions.
        self.assertEqual(theta.shape, (3,))
        self.assertEqual(omega.shape, (3,))
        self.loader.assert_called_once()

    def test_timed_predictions_match_default_return_value(self):
        default_result = infer("unused.pth", num_steps=3)
        theta, omega, elapsed = infer("unused.pth", num_steps=3, return_timing=True)

        self.assertEqual(len(default_result), 2)
        np.testing.assert_array_equal(default_result[0], theta)
        np.testing.assert_array_equal(default_result[1], omega)
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
