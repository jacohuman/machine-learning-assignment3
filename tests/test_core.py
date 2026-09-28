"""
A few basic checks for the equations, training and time-series windows.
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rnn import RNN, train
from datasets import transform, windows, inverse


class ProjectTests(unittest.TestCase):
    def test_recurrence_equations(self):
        # One hidden neuron lets us check each recurrence with scalar arithmetic.
        for kind in ['Elman', 'Jordan', 'Multi']:
            model = RNN(kind, 1)
            with torch.no_grad():
                model.input_layer.weight.fill_(0.7)
                model.input_layer.bias.fill_(0.1)
                model.output_layer.weight.fill_(0.8)
                model.output_layer.bias.fill_(-0.2)
                if kind in ('Elman', 'Multi'):
                    model.hidden_feedback.weight.fill_(0.3)
                if kind in ('Jordan', 'Multi'):
                    model.output_feedback.weight.fill_(-0.4)
            hidden = output = context = 0.0
            for value in [0.2, -0.5, 0.9]:
                context = 0.5 * context + 0.5 * output
                total = 0.7 * value + 0.1
                if kind in ('Elman', 'Multi'):
                    total += 0.3 * hidden
                if kind in ('Jordan', 'Multi'):
                    total -= 0.4 * context
                hidden = math.tanh(total)
                output = 0.8 * hidden - 0.2
            self.assertAlmostEqual(model.predict([[0.2, -0.5, 0.9]])[0], output, places=6)

    def test_training_reduces_error(self):
        x = np.random.default_rng(1).normal(size=(32, 5))
        y = 0.4 * x[:, -1] + 0.2 * x[:, -2]
        for kind in ['Elman', 'Jordan', 'Multi']:
            model = RNN(kind, 8, seed=2)
            before = np.mean((model.predict(x) - y) ** 2)
            train(model, x, y, epochs=150)
            after = np.mean((model.predict(x) - y) ** 2)
            self.assertLess(after, before / 3)

    def test_windows_and_transforms(self):
        values = np.arange(1.0, 101.0)
        for method in ['level', 'seasonal_difference', 'log_difference']:
            raw, z, offset = transform(values, method)
            x, target = windows(z, [40], offset, 0, 1)
            changed = values.copy()
            changed[40:] *= 100  # Change the target and everything after it.
            _, other, _ = transform(changed, method)
            other_x, _ = windows(other, [40], offset, 0, 1)
            np.testing.assert_allclose(x, other_x)
            np.testing.assert_allclose(inverse(target, [40], raw, method), values[[40]])


if __name__ == '__main__':
    unittest.main()
