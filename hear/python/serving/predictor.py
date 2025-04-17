#
# Copyright 2024 Google LLC
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

"""HeAR predictor.

Prepares model input, calls the model, and post-processes the output into the
final response.
"""

import base64
import io
from typing import Any

from absl import logging
from google.oauth2 import credentials
import numpy as np
from scipy import signal
from scipy.io import wavfile

from data_processing import data_processing_lib
from serving.serving_framework import model_runner


_INPUT_ARRAY_KEY = 'input_array'
_INPUT_BYTES_KEY = 'input_bytes'
_GCS_KEY = 'gcs_uri'
_BEARER_TOKEN_KEY = 'bearer_token'
_SAVED_MODEL_DEFAULT_INPUT_KEY = 'x'
_SAVED_MODEL_DEFAULT_OUTPUT_KEY = 'output_0'


KEY_ERROR_MSG = (
    'Request has incorrectly formatted  audio input keys. Must specify exactly '
    'one of `input_bytes`, `gcs_uri`, or `input_array`.'
)


# TODO(b/384088191): Improve error handling and client-facing messaging.
class _PredictorError(Exception):
  """Exception for known predictor errors."""

  def __init__(self, client_message: str):
    super().__init__()
    self.client_message = client_message

  def __str__(self):
    return self.client_message


class Predictor:
  """A predictor for getting embeddings from HeAR."""

  def _process_wav_bytes(self, wav_bytes_b64: bytes | str) -> np.ndarray:
    """Processes wav bytes."""
    if isinstance(wav_bytes_b64, str):
      sample_rate, audio_array = wavfile.read(
          io.BytesIO(base64.b64decode(wav_bytes_b64.encode('utf8')))
      )
    else:
      sample_rate, audio_array = wavfile.read(io.BytesIO(wav_bytes_b64))

    if audio_array.dtype != np.float32:
      if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 2 ** 15
      elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2 ** 31

    if len(audio_array.shape) == 2:
      audio_array = np.mean(audio_array, axis=1)
    if len(audio_array.shape) > 2:
      raise _PredictorError(
          'Audio array must have 1 or 2 dimensions. Got'
          f' {len(audio_array.shape)} dimensions.'
      )
    if sample_rate != 16000:
      num_samples_new = int(len(audio_array) * 16000 / sample_rate)
      audio_array = signal.resample(x=audio_array, num=num_samples_new)

    if audio_array.shape[0] != 32000:
      raise _PredictorError(
          'Audio array must have 32000 samples (2s sampled at 16kHz). Got'
          f' {audio_array.shape[0]} samples.'
      )
    return np.expand_dims(audio_array, axis=0).astype(np.float32)

  def _get_audio_array(self, instance: dict[str, Any]) -> np.ndarray:
    """Gets the audio bytes from a single instance."""
    one_of_is_required_keys = {_INPUT_BYTES_KEY, _GCS_KEY, _INPUT_ARRAY_KEY}
    authorized_keys = one_of_is_required_keys | {_BEARER_TOKEN_KEY}
    available_keys = {key for key in one_of_is_required_keys if key in instance}
    if len(available_keys) != 1 or set(instance.keys()) - authorized_keys:
      raise _PredictorError(KEY_ERROR_MSG)

    if _INPUT_BYTES_KEY in instance:
      wav_bytes = base64.b64decode(instance[_INPUT_BYTES_KEY])
      return self._process_wav_bytes(wav_bytes)

    elif _GCS_KEY in instance:
      creds = (
          credentials.Credentials(token=instance[_BEARER_TOKEN_KEY])
          if _BEARER_TOKEN_KEY in instance
          else None
      )
      gcs_uri = instance[_GCS_KEY]
      logging.info('Retrieving file bytes from GCS: %s', gcs_uri)
      return self._process_wav_bytes(
          data_processing_lib.retrieve_file_bytes_from_gcs(gcs_uri, creds)
      )

    else:
      audio_array = np.array(instance[_INPUT_ARRAY_KEY])
      if len(audio_array.shape) > 1:
        raise _PredictorError(
            'Audio array must have 1 dimension. Got'
            f' {len(audio_array.shape)} dimensions.'
        )
      if audio_array.shape[0] != 32000:
        raise _PredictorError(
            'Audio array must have 32000 samples. Got'
            f' {audio_array.shape[0]} samples.'
        )
      return np.expand_dims(audio_array, axis=0).astype(np.float32)

  def _get_model_input(self, instance: dict[str, Any]) -> dict[str, np.ndarray]:
    """Gets the model input for a single instance."""
    try:
      audio_array = self._get_audio_array(instance)
    except _PredictorError as e:
      raise e
    except Exception as e:
      raise _PredictorError(
          'Failed to retrieve data from request instance.'
      ) from e
    logging.info('Retrieved audio array.')
    return {_SAVED_MODEL_DEFAULT_INPUT_KEY: audio_array}

  def _prepare_response(self, predictions: np.ndarray) -> dict[str, Any]:
    """Prepares the response json for the client."""
    return {'embedding': predictions.tolist()}

  def predict(
      self,
      request: dict[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs model inference on the request instances.

    Args:
      request: The parsed request json to process.
      model: The model runner object to use to call the model.

    Returns:
      The response json which will be returned to the client through the
      Vertex endpoint API.
    """
    predictions: list[dict[str, Any]] = []
    for instance in request['instances']:
      try:
        model_input = self._get_model_input(instance)
        embedding = model.run_model(
            model_input=model_input,
            model_output_key=_SAVED_MODEL_DEFAULT_OUTPUT_KEY,
        )
        # Squash trivial outer dimension
        embedding = np.squeeze(embedding)
        logging.info('Ran inference on model.')
      except _PredictorError as e:
        logging.exception('Failed to get prediction for instance.')
        response = {
            'error': {
                'description': (
                    'Failed to get prediction for instance. Reason:'
                    f' {e.client_message}'
                )
            }
        }
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch-all for any other exceptions that haven't been caught and
        # converted to _PredictorError.
        logging.exception('Failed to get prediction for instance: %s', e)
        response = {
            'error': {
                'description': 'Internal error getting prediction for instance.'
            }
        }
      else:
        response = self._prepare_response(embedding)
        logging.info('Prepared response.')
      predictions.append(response)
    return {'predictions': predictions}
