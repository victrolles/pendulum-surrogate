import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pendulum.evaluation import evaluate


class EvaluationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pointwise_errors_exclude_initial_state_and_prevent_signed_cancellation(self):
        with (
            patch("pendulum.evaluation.simulate", return_value=([999, 1, 2], [999, 10, 20])),
            patch("pendulum.evaluation.infer", return_value=([2, 1], [12, 18], 2.0)) as infer,
            patch("pendulum.evaluation.perf_counter", side_effect=[0, 4])
        ):
            metrics = evaluate("unused.pth", num_steps=2, show=False)

        # Opposite signed errors must not cancel in either error metric.
        self.assertEqual(metrics["mse_theta"], 1.0)
        self.assertEqual(metrics["mse_omega"], 4.0)
        self.assertEqual(metrics["mae_theta"], 1.0)
        self.assertEqual(metrics["mae_omega"], 2.0)
        self.assertEqual(metrics["simulation_time"], 4.0)
        self.assertEqual(metrics["inference_time"], 2.0)
        self.assertEqual(metrics["speedup"], 2.0)
        self.assertTrue(infer.call_args.kwargs["return_timing"])
        for ax in plt.gcf().axes:
            for line in ax.lines:
                np.testing.assert_allclose(line.get_xdata(), [0.02, 0.04])

    def test_custom_grid_can_compare_a_single_transition(self):
        with (
            patch("pendulum.evaluation.simulate", return_value=([1, 2], [3, 4])) as simulate,
            patch("pendulum.evaluation.infer", return_value=([2], [4], 1.0)) as infer
        ):
            metrics = evaluate(
                "custom.pth", num_steps=1, hidden_size=8, num_hidden_layers=2,
                l=2.0, m=3.0, b=0.2, theta0=1.0, omega0=3.0,
                t=np.array([5.0, 5.1]), show=False
            )

        self.assertEqual(metrics["mse_theta"], 0.0)
        self.assertEqual(metrics["mse_omega"], 0.0)
        self.assertEqual(metrics["mae_theta"], 0.0)
        self.assertEqual(metrics["mae_omega"], 0.0)
        self.assertEqual(infer.call_args.kwargs["num_steps"], 1)
        self.assertEqual(infer.call_args.kwargs["hidden_size"], 8)
        for parameter in ("l", "m", "b", "theta0", "omega0"):
            self.assertEqual(simulate.call_args.kwargs[parameter], infer.call_args.kwargs[parameter])
        np.testing.assert_allclose(plt.gcf().axes[0].lines[0].get_xdata(), [5.1])

    def test_absolute_errors_handle_zero_crossings_and_zero_reference(self):
        with (
            patch("pendulum.evaluation.simulate", return_value=([999, -1, 0, 1], [999, 0, 0, 0])),
            patch("pendulum.evaluation.infer", return_value=([-2, 0, 2], [0, 0, 0], 1.0))
        ):
            metrics = evaluate("unused.pth", num_steps=3, show=False)
        self.assertAlmostEqual(metrics["mae_theta"], 2 / 3)
        self.assertEqual(metrics["mae_omega"], 0.0)

    def test_zero_reference_does_not_amplify_absolute_error(self):
        with (
            patch("pendulum.evaluation.simulate", return_value=([0, 0], [0, 0])),
            patch("pendulum.evaluation.infer", return_value=([1], [0], 1.0))
        ):
            metrics = evaluate("unused.pth", num_steps=1, show=False)
        self.assertEqual(metrics["mae_theta"], 1.0)
        self.assertEqual(metrics["mae_omega"], 0.0)

    def test_errors_are_measured_in_variable_units(self):
        with (
            patch("pendulum.evaluation.simulate", return_value=([0, 1], [0, -1])),
            patch("pendulum.evaluation.infer", return_value=([3], [-3], 1.0))
        ):
            metrics = evaluate("unused.pth", num_steps=1, show=False)
        self.assertEqual(metrics["mae_theta"], 2.0)
        self.assertEqual(metrics["mae_omega"], 2.0)

    def test_invalid_comparison_grids_fail_before_execution(self):
        cases = [
            {"num_steps": 0},
            {"num_steps": 1.5},
            {"num_steps": 2, "t": [0, 0.02]},
            {"num_steps": 2, "t": [0, 0.02, 0.05]},
            {"num_steps": 2, "t": [0, 0.02, 0.01]},
            {"num_steps": 2, "t": [0, 0.02, np.nan]}
        ]
        with patch("pendulum.evaluation.simulate") as simulate, patch("pendulum.evaluation.infer") as infer:
            for arguments in cases:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    evaluate("unused.pth", show=False, **arguments)
            simulate.assert_not_called()
            infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
