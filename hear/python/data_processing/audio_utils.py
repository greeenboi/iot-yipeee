#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for audio processing."""

import math
from typing import Callable

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


def _enclosing_power_of_two(value: int) -> int:
  """Calculates the smallest power of 2 greater than or equal to `value`."""
  return int(2 ** math.ceil(math.log2(value))) if value > 0 else 1


def _compute_stft(
    signals: torch.Tensor,
    frame_length: int,
    frame_step: int,
    fft_length: int | None = None,
    window_fn: Callable[[int], torch.Tensor] | None = torch.hann_window,
    pad_end: bool = True,
) -> torch.Tensor:
  """Computes the Short-time Fourier Transform of `signals`.

  Args:
    signals: A `[..., samples]` `torch.Tensor` of real-valued signals.
    frame_length: The window length in samples.
    frame_step: The number of samples to step.
    fft_length: The size of the FFT to apply. If not provided, uses the smallest
      power of 2 enclosing `frame_length`.
    window_fn: A callable that takes a window length and returns a
      `[window_length]` `torch.Tensor` of samples. If set to `None`, no
      windowing is used.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.

  Returns:
    A `[..., frames, fft_unique_bins]` `torch.Tensor` of `complex64`
    STFT values where `fft_unique_bins` is `fft_length // 2 + 1` (the unique
    components of the FFT).

  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_step` is not scalar.
  """
  if signals.ndim < 1:
    raise ValueError(
        f'Input signals must have rank at least 1, got rank {signals.ndim}'
    )
  if not isinstance(frame_length, int):
    raise ValueError(
        f'frame_length must be an integer scalar, got type {type(frame_length)}'
    )
  if not isinstance(frame_step, int):
    raise ValueError(
        f'frame_step must be an integer scalar, got type {type(frame_step)}'
    )

  if fft_length is None:
    fft_length = _enclosing_power_of_two(frame_length)
  elif not isinstance(fft_length, int):
    raise ValueError(
        'fft_length must be an integer scalar or None, got type'
        f' {type(fft_length)}'
    )

  if pad_end:
    n_frames = (
        math.ceil(signals.shape[-1] / frame_step)
        if signals.shape[-1] > 0
        else 0
    )
    padded_length = (
        max(0, (n_frames - 1) * frame_step + frame_length)
        if n_frames > 0
        else frame_length
    )
    padding_needed = max(0, padded_length - signals.shape[-1])
    if padding_needed > 0:
      signals = F.pad(signals, (0, padding_needed))

  framed_signals = signals.unfold(-1, frame_length, frame_step)

  if framed_signals.shape[-2] == 0:
    return torch.empty(
        *signals.shape[:-1],
        0,
        fft_length // 2 + 1,
        dtype=torch.complex64,
        device=signals.device,
    )

  # Optionally window the framed signals.
  if window_fn is not None:
    window = (
        window_fn(frame_length)
        .to(framed_signals.device)
        .to(framed_signals.dtype)
    )  # Ensure window is on the same device and dtype
    framed_signals = framed_signals * window

  # torch.fft.rfft produces the (fft_length/2 + 1) unique components of the
  # FFT of the real windowed signals in framed_signals.
  return torch.fft.rfft(framed_signals, n=fft_length, dim=-1)


def _ema(
    inputs: torch.Tensor,
    num_channels: int,
    smooth_coef: float,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
  """Exponential Moving Average (EMA).

  Args:
    inputs: A 3D tensor of shape (batch_size, timesteps, input_dim). This is the
      input sequence to the EMA RNN.
    num_channels: The number of channels/units in the EMA.
    smooth_coef: The smoothing coefficient (alpha) for the EMA.
    initial_state: (Optional) A 2D tensor of shape (batch_size, num_channels)
      representing the initial state for the EMA. If None, it defaults to zeros.

  Returns:
    A 3D tensor of shape (batch_size, timesteps, num_channels) representing
    the EMA output sequence.
  """
  batch_size, timesteps, _ = inputs.shape

  if initial_state is None:
    ema_state = torch.zeros(
        (batch_size, num_channels), dtype=torch.float32, device=inputs.device
    )
  else:
    ema_state = initial_state

  identity_kernel_gain = smooth_coef
  identity_recurrent_gain = 1.0 - smooth_coef

  identity_kernel = (
      torch.eye(num_channels, dtype=torch.float32, device=inputs.device)
      * identity_kernel_gain
  )
  identity_recurrent_kernel = (
      torch.eye(num_channels, dtype=torch.float32, device=inputs.device)
      * identity_recurrent_gain
  )

  output_sequence = []

  start = initial_state is not None
  if start:
    output_sequence.append(ema_state)
  for t in range(start, timesteps):
    current_input = inputs[:, t, :]  # Shape (batch_size, num_channels)

    # EMA update formula:
    output = torch.matmul(current_input, identity_kernel) + torch.matmul(
        ema_state, identity_recurrent_kernel
    )

    ema_state = output
    output_sequence.append(output)

  # Shape (batch_size, timesteps, num_channels)
  output_sequence = torch.stack(output_sequence, dim=1)
  return output_sequence


def _pcen_function(
    inputs: torch.Tensor,
    num_channels: int = 128,
    alpha: float = 0.8,
    smooth_coef: float = 0.04,
    delta: float = 2.0,
    root: float = 2.0,
    floor: float = 1e-8,
) -> torch.Tensor:
  """Per-Channel Energy Normalization as a function.

  This applies a fixed normalization by an exponential moving
  average smoother, and a compression.
  See https://arxiv.org/abs/1607.05666 for more details.

  Args:
    inputs: A `[..., num_channels]` `torch.Tensor` of input signals.
    num_channels: Number of channels
    alpha: Exponent of EMA smoother
    smooth_coef: Smoothing coefficient of EMA
    delta: Bias added before compression
    root: One over exponent applied for compression (r in the paper)
    floor: Offset added to EMA smoother

  Returns:
    A `[..., num_channels]` `torch.Tensor` of PCEN values.
  """
  alpha_param = torch.ones(num_channels) * alpha
  delta_param = torch.ones(num_channels) * delta
  root_param = torch.ones(num_channels) * root

  alpha_param = (
      torch.minimum(alpha_param, torch.ones_like(alpha_param))
      .to(inputs.device)
      .to(inputs.dtype)
  )
  root_param = (
      torch.maximum(root_param, torch.ones_like(root_param))
      .to(inputs.device)
      .to(inputs.dtype)
  )
  ema_smoother = _ema(
      inputs,
      num_channels=num_channels,
      smooth_coef=smooth_coef,
      # Handle cases where input is 2D or 3D
      initial_state=inputs[:, 0] if inputs.ndim > 1 else None,
  )

  one_over_root = 1.0 / root_param
  output = (
      inputs / (floor + ema_smoother) ** alpha_param + delta_param
  ) ** one_over_root - delta_param**one_over_root
  return output


def _hertz_to_mel(frequencies_hertz: torch.Tensor) -> torch.Tensor:
  """Scale filter frequencies to mel scale.

  https://en.wikipedia.org/wiki/Mel_scale

  Args:
    frequencies_hertz: A float Tensor of frequencies in Hertz.

  Returns:
    A Tensor of the same shape but in mel-scale.
  """
  return 2595.0 * torch.log10(1.0 + frequencies_hertz / 700.0)


def _linear_to_mel_weight_matrix(
    num_mel_bins: int = 128,
    num_spectrogram_bins: int = 201,
    sample_rate: float = 16000,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 8000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Returns a matrix to warp linear scale spectrograms to the mel scale.

  This function is a PyTorch version of the TensorFlow function with the same
  name. See the TensorFlow documentation for detailed explanation and usage
  https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix

  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data.
    sample_rate: Samples per second of the input signal used to create the
      spectrogram.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.
    upper_edge_hertz: The desired top edge of the highest frequency band.
    dtype: The torch.dtype of the result matrix. Must be a floating point type.

  Returns:
    A torch.Tensor of shape `[num_spectrogram_bins, num_mel_bins]`.

  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
  """

  if num_mel_bins <= 0:
    raise ValueError(f'num_mel_bins must be positive. Got: {num_mel_bins}.')
  if num_spectrogram_bins <= 0:
    raise ValueError(
        f'num_spectrogram_bins must be positive. Got: {num_spectrogram_bins}.'
    )
  if sample_rate <= 0:
    raise ValueError(f'sample_rate must be positive. Got: {sample_rate}.')
  if lower_edge_hertz < 0.0:
    raise ValueError(
        f'lower_edge_hertz must be non-negative. Got: {lower_edge_hertz}.'
    )
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError(
        'lower_edge_hertz must be smaller than upper_edge_hertz. Got: '
        f'lower_edge_hertz={lower_edge_hertz}, '
        f'upper_edge_hertz={upper_edge_hertz}.'
    )
  if upper_edge_hertz > sample_rate / 2.0:
    raise ValueError(
        'upper_edge_hertz must not be larger than the Nyquist frequency'
        f'({sample_rate / 2.0}). Got: upper_edge_hertz={upper_edge_hertz}.'
    )

  sample_rate_tensor = torch.tensor(sample_rate, dtype=dtype)
  lower_edge_hertz_tensor = torch.tensor(lower_edge_hertz, dtype=dtype)
  upper_edge_hertz_tensor = torch.tensor(upper_edge_hertz, dtype=dtype)
  zero = torch.tensor(0.0, dtype=dtype)

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate_tensor / 2.0
  linear_frequencies = torch.linspace(
      zero, nyquist_hertz, num_spectrogram_bins, dtype=dtype
  )[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies).unsqueeze(1)

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge).
  # The center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = torch.linspace(
      _hertz_to_mel(lower_edge_hertz_tensor),
      _hertz_to_mel(upper_edge_hertz_tensor),
      num_mel_bins + 2,
      dtype=dtype,
  )
  # Create frames of size 3 with stride 1
  band_edges_mel = band_edges_mel.unfold(0, 3, 1)

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel = band_edges_mel[:, 0].unsqueeze(0)
  center_mel = band_edges_mel[:, 1].unsqueeze(0)
  upper_edge_mel = band_edges_mel[:, 2].unsqueeze(0)

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel
  )
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel
  )

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = torch.maximum(
      zero, torch.minimum(lower_slopes, upper_slopes)
  )

  # Re-add the zeroed lower bins we sliced out above.
  return F.pad(
      mel_weights_matrix, (0, 0, bands_to_zero, 0), mode='constant', value=0.0
  )


def _mel_pcen(
    x: torch.Tensor,
) -> torch.Tensor:
  """Melspec followed by pcen."""
  x = x.float()
  # Scale to -1, 1 range
  x -= torch.min(x)
  x = x / (torch.max(x) + 1e-8)
  x = (x * 2) - 1

  frame_length = 16 * 25
  frame_step = 160

  stft = _compute_stft(
      x,
      frame_length=frame_length,
      fft_length=frame_length,
      frame_step=frame_step,
      window_fn=torch.hann_window,
      pad_end=True,
  )
  spectrograms = torch.square(torch.abs(stft))

  mel_transform = _linear_to_mel_weight_matrix()
  mel_spectrograms = torch.matmul(spectrograms, mel_transform)
  return _pcen_function(mel_spectrograms)


def _torch_resize_bilinear_tf_compat(images, size):
  """PyTorch implementation of tf.image.resize.

  Internally matches the numerical output of:
  ```
    tf.image.resize(
        # TF input needs HWC/BHWC format
        tf_permuted_input,
        size,
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
        antialias=False
    )
  ```

  Args:
      images: Input tensor, shape [C, H, W] or [B, C, H, W]. Should be float32
        or convertible to float32 for numerical matching.
      size: Target size as [new_height, new_width].

  Returns:
      Resized tensor, shape [C, new_H, new_W] or [B, C, new_H, new_W], dtype
        torch.float32.
  """
  original_dims = images.dim()
  new_height, new_width = size

  if original_dims not in [3, 4]:
    raise ValueError('Input tensor must be 3D [C, H, W] or 4D [B, C, H, W]')

  if len(size) != 2:
    raise ValueError(
        'size must be a tuple or list of 2 integers (new_height, new_width)'
    )

  images = images.to(torch.float32)

  was_3d = False
  if original_dims == 3:
    images = images.unsqueeze(0)  # Shape: [1, C, H, W]
    was_3d = True

  resized_images_bchw = F.interpolate(
      images,  # Shape: [B, C, H, W]
      size=(new_height, new_width),
      mode='bilinear',
      align_corners=False,
      antialias=False,
  )
  # Output shape: [B, C, new_H, new_W]

  # Remove batch dimension if original input was 3D (1CNHWC -> CNHWC)
  if was_3d:
    # Shape: [C, new_H, new_W]
    resized_images_bchw = resized_images_bchw.squeeze(0)

  return resized_images_bchw


def preprocess_audio(audio: torch.Tensor) -> torch.Tensor:
  """Preprocesses audio.

  Args:
    audio: A `[..., samples]` `torch.Tensor` of real-valued signals. Represents
      batched audio clips of duration 2s sampled at 16kHz.

  Returns:
    A `[..., 1, 192, 128]` `torch.Tensor` of mel-pcen values.
  """
  if audio.ndim != 2:
    raise ValueError(
        f'Input audio must have rank 2, got rank {audio.ndim}'
    )
  if audio.shape[1] < 32000:
    n = 32000 - audio.shape[1]
    audio = torch.nn.functional.pad(audio, pad=(0, n), mode='constant', value=0)
  elif audio.shape[1] > 32000:
    raise ValueError(
        f'Input audio must have 32000 samples, got {audio.shape[1]}'
    )
  spectrogram = _mel_pcen(audio)
  # Add a channel dimension. Torch images have format [B, C, H, W].
  spectrogram = torch.unsqueeze(spectrogram, dim=1)
  return _torch_resize_bilinear_tf_compat(spectrogram, size=(192, 128))


def resample_audio_and_convert_to_mono(
    audio_array: np.ndarray,
    sampling_rate: float,
    new_sampling_rate: float,
) -> np.ndarray:
  """Resamples an audio array to 16kHz and converts it to mono.

  Args:
    audio_array: A numpy array representing the audio data.
    sampling_rate: The original sampling rate of the audio.
    new_sampling_rate: Target sampling rate.

  Returns:
    resampled_audio_mono: A numpy array representing the resampled mono audio at
    16kHz.
  """
  # Convert to mono if it's multi-channel
  if audio_array.ndim > 1:
    audio_mono = np.mean(audio_array, axis=1)
  else:
    audio_mono = audio_array

  # Resample
  original_sample_count = audio_mono.shape[0]
  new_sample_count = int(
      round(original_sample_count * (new_sampling_rate / sampling_rate))
  )
  resampled_audio_mono = signal.resample(audio_mono, new_sample_count)

  return resampled_audio_mono
