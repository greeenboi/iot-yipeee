# Background Audio Processor

This is a standalone Python script that continuously records audio in 30-second snippets, processes them through the cough classification model, and outputs the results.

## Features

- Continuously records 30-second audio snippets from the default microphone
- Processes each snippet through the cough classification model
- Outputs prediction results as simple print statements
- Automatically cleans up processed audio files
- Runs in the background until manually stopped

## Requirements

In addition to the project's existing requirements, you'll need PyAudio for audio recording:

```bash
pip install pyaudio
```

Note: Installing PyAudio might require additional system dependencies:

- **Windows**: Should work with pip install
- **macOS**: `brew install portaudio` before pip install
- **Linux**: `sudo apt-get install python3-pyaudio` or `sudo apt-get install portaudio19-dev` before pip install

## Usage

1. Make sure the model file `cough_classification_model.pkl` is in the same directory as the script
2. Run the script:

```bash
python background_audio_processor.py
```

3. The script will start recording audio in 30-second snippets and processing them
4. Results will be printed to the console
5. Press Ctrl+C to stop the script

## How It Works

1. The script creates a temporary directory to store audio snippets
2. It starts a background thread to periodically clean up old audio files
3. The main loop:
   - Records 30 seconds of audio
   - Saves the audio to a temporary file
   - Extracts features from the audio
   - Runs the features through the model
   - Prints the prediction and probabilities
   - Repeats

4. When the script is stopped (Ctrl+C), it performs a final cleanup of all temporary files

## Customization

You can modify the following constants at the top of the script:

- `RECORD_SECONDS`: Duration of each audio snippet (default: 30)
- `RATE`: Sample rate for audio recording (default: 22050)
- `CHANNELS`: Number of audio channels (default: 1 for mono)
- `FORMAT`: Audio format (default: 16-bit PCM)
- `TEMP_DIR`: Directory for temporary audio files

## Troubleshooting

If you encounter issues with audio recording:

1. Check that your microphone is properly connected and set as the default recording device
2. Try adjusting the audio recording parameters (RATE, CHANNELS, FORMAT)
3. Make sure you have the necessary permissions to access the microphone

If you encounter issues with model prediction:

1. Verify that the model file exists and is in the correct location
2. Check that the model was trained with the same feature extraction process