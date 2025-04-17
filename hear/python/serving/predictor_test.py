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

import base64
import io
from unittest import mock

import numpy as np
from scipy.io import wavfile

from absl.testing import absltest
from absl.testing import parameterized
from data_processing import data_processing_lib
from serving.serving_framework import model_runner
from serving import predictor


_WAV_CONTENT = np.arange(32000).astype(np.float32)


def _create_dummy_wav_as_b64() -> str:
  """Creates a dummy WAV file.

  Returns:
      The bytes of the WAV file encoded with base64.
  """
  bytes_io = io.BytesIO()
  wavfile.write(bytes_io, 16000, _WAV_CONTENT)
  return base64.b64encode(bytes_io.getvalue()).decode("utf-8")


@mock.patch.object(model_runner, "ModelRunner", autospec=True)
class PredictorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._request_instance = {
        "instances": [{
            "gcs_uri": "gs://bucket/file.wav",
            "bearer_token": "my_token",
        }]
    }

  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      return_value=_create_dummy_wav_as_b64(),
  )
  def test_predict_model_called_with_correct_input(
      self, unused_mock_retrieve, mock_model_runner
  ):
    mock_model_runner = mock.Mock()
    predictor_instance = predictor.Predictor()

    with mock.patch.object(
        predictor_instance, "_process_wav_bytes"
    ) as mock_process_wav_bytes:
      mock_process_wav_bytes.return_value = _WAV_CONTENT

      predictor_instance.predict(
          request=self._request_instance,
          model=mock_model_runner,
      )

      mock_model_runner.run_model.assert_called_once_with(
          # Use ANY as a placeholder because '==' is not supported for numpy
          # arrays.
          model_input=mock.ANY,
          model_output_key=predictor._SAVED_MODEL_DEFAULT_OUTPUT_KEY,
      )
      _, actual_kwargs = mock_model_runner.run_model.call_args
      self.assertSetEqual(
          set(actual_kwargs["model_input"].keys()),
          {predictor._SAVED_MODEL_DEFAULT_INPUT_KEY},
      )
      np.testing.assert_array_equal(
          actual_kwargs["model_input"][
              predictor._SAVED_MODEL_DEFAULT_INPUT_KEY
          ],
          _WAV_CONTENT,
          strict=True,
      )

  @mock.patch.object(predictor.Predictor, "_get_model_input", autospec=True)
  def test_predict_returns_correct_response(
      self, unused_mock_model_input, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = np.array([[1, 2, 3]])
    response = predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    self.assertEqual(response, {"predictions": [{"embedding": [1, 2, 3]}]})

  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      side_effect=ValueError("some retrieval error"),
  )
  def test_predict_retrieve_audio_data_failure_returns_error_response(
      self, unused_retrieve, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request=self._request_instance,
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Failed to retrieve data"
        " from request instance.",
    )

  def test_predict_without_audio_input_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={"instances": [{}]},
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason:"
        f" {predictor.KEY_ERROR_MSG}",
    )

  def test_predict_with_multiple_audio_inputs_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={
            "instances": [{
                "gcs_uri": "gs://bucket/file.dcm",
                "input_bytes": "c29tZV9ieXRlcw==",
            }]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason:"
        f" {predictor.KEY_ERROR_MSG}",
    )

  def test_predict_with_multidim_audio_array_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={
            "instances": [{
                "input_array": [_WAV_CONTENT, _WAV_CONTENT],
            }]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Audio array must have 1"
        " dimension. Got 2 dimensions.",
    )

  def test_predict_with_audio_array_incorrect_samples_returns_error_response(
      self, mock_model_runner
  ):
    response = predictor.Predictor().predict(
        request={
            "instances": [{
                "input_array": np.arange(32001).astype(np.float32),
            }]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Audio array must have"
        " 32000 samples. Got 32001 samples.",
    )

  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      side_effect=[
          KeyError("some retrieval error"),
          _create_dummy_wav_as_b64(),
      ],
  )
  def test_predict_with_multiple_request_instances_returns_correct_response(
      self, unused_mock_retrieve, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = [_WAV_CONTENT]
    response = predictor.Predictor().predict(
        request={
            "instances": [
                {
                    "gcs_uri": "gs://bucket/file1.wav",
                    "bearer_token": "my_token",
                },
                {
                    "gcs_uri": "gs://bucket/file2.wav",
                    "bearer_token": "my_token",
                },
            ]
        },
        model=mock_model_runner,
    )
    self.assertEqual(
        response["predictions"][0]["error"]["description"],
        "Failed to get prediction for instance. Reason: Failed to retrieve data"
        " from request instance.",
    )
    np.testing.assert_array_equal(
        np.asarray(response["predictions"][1]["embedding"]), _WAV_CONTENT
    )

  @mock.patch.object(
      data_processing_lib,
      "retrieve_file_bytes_from_gcs",
      autospec=True,
      side_effect=[
          _create_dummy_wav_as_b64(),
      ],
  )
  def test_predict_gcs_no_bearer_token_returns_correct_response(
      self, unused_mock_retrieve, mock_model_runner
  ):
    mock_model_runner.run_model.return_value = [_WAV_CONTENT]
    response = predictor.Predictor().predict(
        request={
            "instances": [
                {
                    "gcs_uri": "gs://bucket/file.wav",
                },
            ]
        },
        model=mock_model_runner,
    )
    np.testing.assert_array_equal(
        np.asarray(response["predictions"][0]["embedding"]), _WAV_CONTENT
    )


class TestProcessWavBytes(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.predictor = predictor.Predictor()

  def _create_wav_data(self, sample_rate, data):
    """Helper function to create in-memory wav data."""
    with io.BytesIO() as f:
      wavfile.write(f, sample_rate, data)
      return f.getvalue()

  def test_already_float32(self):
    sample_rate = 16000
    audio_data = np.zeros(32000, dtype=np.float32)
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    np.testing.assert_array_equal(result, np.expand_dims(audio_data, axis=0))
    self.assertEqual(result.dtype, np.float32)

  def test_int16_conversion(self):
    sample_rate = 16000
    audio_data = np.random.randint(
        -(2**15) + 1, 2**15 - 1, size=32000, dtype=np.int16
    )
    expected_output = np.expand_dims(
        audio_data.astype(np.float32) / 2**15, axis=0
    )
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    np.testing.assert_array_equal(result, expected_output)
    self.assertEqual(result.dtype, np.float32)

  def test_int32_conversion(self):
    sample_rate = 16000
    audio_data = np.random.randint(
        -(2**31) + 1, 2**31 - 1, size=32000, dtype=np.int32
    )
    expected_output = np.expand_dims(
        audio_data.astype(np.float32) / 2**31, axis=0
    )
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    np.testing.assert_array_equal(result, expected_output)
    self.assertEqual(result.dtype, np.float32)

  def test_stereo_to_mono(self):
    sample_rate = 16000
    stereo_data = np.random.randint(0, 1000, size=(32000, 2), dtype=np.int16)
    expected_mono = np.expand_dims(stereo_data.mean(axis=-1) / 2**15, axis=0)
    wav_bytes = self._create_wav_data(sample_rate, stereo_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    np.testing.assert_array_equal(result, expected_mono)
    self.assertEqual(result.shape, (1, 32000))

  @parameterized.named_parameters(
      dict(
          testcase_name="higher_rate",
          original_sample_rate=32000,
      ),
      dict(
          testcase_name="higher_rate_44100",
          original_sample_rate=44100,
      ),
      dict(
          testcase_name="lower_rate",
          original_sample_rate=8000,
      ),
      dict(
          testcase_name="lower_rate_7000",
          original_sample_rate=7000,
      ),
  )
  def test_resampling(self, original_sample_rate):
    original_duration_seconds = 2
    audio_data = np.zeros(
        original_sample_rate * original_duration_seconds, dtype=np.float32
    )
    wav_bytes = self._create_wav_data(original_sample_rate, audio_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    self.assertEqual(result.shape, (1, 32000))

  def test_correct_sample_rate_no_resampling(self):
    sample_rate = 16000
    audio_data = np.zeros(32000, dtype=np.float32)
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    with mock.patch("scipy.signal.resample") as mock_resample:
      self.predictor._process_wav_bytes(wav_bytes)
      mock_resample.assert_not_called()

  def test_resampling_not_32000_samples(self):
    original_sample_rate = 8000
    target_sample_rate = 16000
    original_num_samples = 1000
    audio_data = np.zeros(original_num_samples, dtype=np.float32)
    wav_bytes = self._create_wav_data(original_sample_rate, audio_data)

    expected_num_samples_after_resample = int(
        original_num_samples * target_sample_rate / original_sample_rate
    )

    with mock.patch("scipy.io.wavfile.read") as mock_wavfile_read:
      mock_wavfile_read.return_value = (original_sample_rate, audio_data)

      with mock.patch("scipy.signal.resample") as mock_resample:
        mock_resample.return_value = np.zeros(
            expected_num_samples_after_resample
        )

        with self.assertRaisesRegex(
            predictor._PredictorError, "Audio array must have 32000 samples"
        ):
          self.predictor._process_wav_bytes(wav_bytes)

  def test_correct_length(self):
    sample_rate = 16000
    audio_data = np.zeros(32000, dtype=np.float32)
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    result = self.predictor._process_wav_bytes(wav_bytes)
    self.assertEqual(result.shape, (1, 32000))

  def test_incorrect_length_raises_error(self):
    sample_rate = 16000
    audio_data = np.zeros(16001, dtype=np.float32)
    wav_bytes = self._create_wav_data(sample_rate, audio_data)
    with self.assertRaisesRegex(
        predictor._PredictorError, "Audio array must have 32000 samples"
    ):
      self.predictor._process_wav_bytes(wav_bytes)

  def test_wavfile_read_raises_exception(self):
    wav_bytes = b"invalid wav data"
    with self.assertRaisesRegex(
        Exception,
        "File format b'inva' not understood. Only .*",
    ):
      self.predictor._process_wav_bytes(wav_bytes)


if __name__ == "__main__":
  absltest.main()
