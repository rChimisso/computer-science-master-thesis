import unittest

import torch

from src.models.temporal_reference import TemporalLSTM, masked_class_max
from src.models.training_support import configure_device

class NeuralTests(unittest.TestCase):
  """Protect temporal neural loss and locked dropout."""

  def setUp(self):
    """Seed bounded CPU checks and restore the caller's thread budget afterward."""
    self.addCleanup(torch.set_num_threads, torch.get_num_threads())
    configure_device(42, 'cpu', 1)

  def test_classwise_max_and_finite_gradients(self) -> None:
    """Select maxima independently per class and propagate only finite gradients."""
    logits = torch.tensor([[[1.0, 4.0], [3.0, 2.0], [99.0, 99.0]]], requires_grad=True)
    pooled = masked_class_max(logits, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 4.0]]))
    pooled.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]))
    model = TemporalLSTM(4, hidden_size=8)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 5, 4), torch.tensor([2, 3, 5])), torch.tensor([0, 1, 2]))
    loss.backward()
    self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

  def test_locked_dropout_and_reproducibility(self) -> None:
    """Use one dropout mask across time and repeat a fixed CPU training step."""
    model = TemporalLSTM(16, hidden_size=8, input_dropout=0.5)
    captured = []
    def capture_inputs(module: torch.nn.Module, arguments: tuple) -> None:
      """Capture the recurrent input for a temporal dropout assertion.

      :param module: Invoked recurrent layer.
      :param arguments: Positional forward inputs.
      """
      captured.append(arguments[0].detach().clone())
    hook = model.recurrent.register_forward_pre_hook(capture_inputs)
    inputs, lengths = torch.ones(4, 5, 16), torch.full((4,), 5)
    configure_device(7, "cpu", 1)
    first = model(inputs, lengths)
    configure_device(7, "cpu", 1)
    second = model(inputs, lengths)
    hook.remove()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(captured[0][:, :1].expand_as(captured[0]), captured[0], atol=0, rtol=0)
    self.assertGreater(int((captured[0] == 0).sum()), 0)
