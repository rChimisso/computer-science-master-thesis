import math

import torch
from torch import nn

def build_network(channels: int, classes: int, model: str, settings: dict, device: torch.device) -> nn.Module:
  """Construct a temporal model independently of fitting and optimizer state.

  :param channels: Input feature count.
  :param classes: Dataset class count.
  :param model: LSTM or Transformer.
  :param settings: Resolved architectural and regularization values.
  :param device: Explicit execution device.
  :return: Initialized model on the requested device.
  """
  if model == "lstm":
    return TemporalLSTM(channels, settings["hidden_size"], classes, settings["input_dropout"], settings["output_dropout"]).to(device)
  if model == "transformer":
    return TemporalTransformer(channels, classes, **{key: settings[key] for key in ("width", "layers", "heads", "feedforward_size", "dropout")}).to(device)
  raise ValueError("Unknown temporal model")

def valid_time_mask(inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
  """Construct a valid-time mask for padded sequences.

  :param inputs: Tensor shaped batch, time, channel.
  :param lengths: Positive valid lengths on the input device.
  :return: Boolean batch-time mask.
  :raises ValueError: If input dimensions or lengths are invalid.
  """
  if inputs.ndim != 3 or lengths.shape != (len(inputs),):
    raise ValueError("Temporal inputs and lengths are incompatible")
  if torch.any(lengths < 1) or torch.any(lengths > inputs.shape[1]):
    raise ValueError("Temporal lengths lie outside the padded tensor")
  return torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths[:, None]

def masked_class_max(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Reduce each class over valid time steps independently.

  :param logits: Batch-time-class logits before softmax.
  :param mask: Boolean valid-time mask.
  :return: One maximum logit per sample and class.
  """
  return logits.masked_fill(~mask[:, :, None], -torch.inf).amax(dim=1)

class TemporalLSTM(nn.Module):
  """Fused unidirectional LSTM with explicit valid-time classification.

  Input dropout is locked across time. Output dropout does not implement the recurrent dropout of the original Keras reference.
  """

  input_dropout: float
  """Probability of dropping each sample's input channel for the whole sequence."""
  recurrent: nn.LSTM
  """Fused fixed-width trainable recurrent layer."""
  output_dropout: nn.Dropout
  """Dropout on recurrent outputs before the classifier."""
  classifier: nn.Linear
  """Linear per-time-step class projection."""

  def __init__(self, input_dimension: int, hidden_size: int = 128, classes: int = 20, input_dropout: float = 0.2, output_dropout: float = 0.2):
    """Initialize the fast paper-informed temporal model.

    :param input_dimension: Number of cochlear features.
    :param hidden_size: Number of recurrent units.
    :param classes: Number of target classes.
    :param input_dropout: Time-locked input dropout probability.
    :param output_dropout: Recurrent-output dropout probability.
    """
    super().__init__()
    self.input_dropout = input_dropout
    self.recurrent = nn.LSTM(input_dimension, hidden_size, batch_first=True)
    self.output_dropout = nn.Dropout(output_dropout)
    self.classifier = nn.Linear(hidden_size, classes)

  def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Compute classwise maximum logits without padding contributions.

    :param inputs: Padded batch-time-channel tensor.
    :param lengths: Valid sequence lengths.
    :return: Batch-class logits.
    """
    mask = valid_time_mask(inputs, lengths)
    values = torch.where(mask[:, :, None], inputs, torch.zeros((), device=inputs.device, dtype=inputs.dtype))
    if self.training and self.input_dropout:
      keep = 1.0 - self.input_dropout
      dropout_mask = torch.empty((len(values), 1, values.shape[2]), device=values.device, dtype=values.dtype).bernoulli_(keep) / keep
      values = values * dropout_mask
    sequential, _ = self.recurrent(values)
    return masked_class_max(self.classifier(self.output_dropout(sequential)), mask)

class TemporalTransformer(nn.Module):
  """Compact positional temporal encoder with masked mean pooling."""

  projection: nn.Linear
  """Learned input projection into the encoder width."""
  encoder: nn.TransformerEncoder
  """Attention encoder with explicit padding masking."""
  classifier: nn.Linear
  """Pooled-state classification head."""
  width: int
  """Even positional encoding width."""

  def __init__(self, input_dimension: int, classes: int = 20, width: int = 64, layers: int = 2, heads: int = 4, feedforward_size: int = 128, dropout: float = 0.1):
    """Initialize the compact Transformer reference.

    :param input_dimension: Input feature count.
    :param classes: Target class count.
    :param width: Encoder embedding dimension.
    :param layers: Number of encoder blocks.
    :param heads: Attention heads per block.
    :param feedforward_size: Internal feed-forward width.
    :param dropout: Encoder dropout probability.
    """
    super().__init__()
    if width % 2:
      raise ValueError("Positional encoding requires an even width")
    self.width = width
    self.projection = nn.Linear(input_dimension, width)
    layer = nn.TransformerEncoderLayer(width, heads, feedforward_size, dropout, batch_first=True)
    self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    self.classifier = nn.Linear(width, classes)

  def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Classify a sequence using valid states only.

    :param inputs: Padded batch-time-feature tensor.
    :param lengths: Valid lengths.
    :return: Batch-class logits.
    """
    mask = valid_time_mask(inputs, lengths)
    values = torch.where(mask[:, :, None], inputs, torch.zeros((), device=inputs.device, dtype=inputs.dtype))
    projected = self.projection(values)
    positions = torch.arange(inputs.shape[1], device=inputs.device, dtype=projected.dtype)[:, None]
    frequencies = torch.exp(torch.arange(0, self.width, 2, device=inputs.device, dtype=projected.dtype) * (-math.log(10_000.0) / self.width))
    encoding = torch.zeros((inputs.shape[1], self.width), device=inputs.device, dtype=projected.dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    encoded = self.encoder(projected + encoding[None], src_key_padding_mask=~mask)
    pooled = torch.where(mask[:, :, None], encoded, 0.0).sum(dim=1) / lengths[:, None]
    return self.classifier(pooled)
