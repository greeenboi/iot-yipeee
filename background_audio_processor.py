import os
import time
import pickle
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyaudio
import wave
import tempfile
import shutil
import datetime
import threading
import RPi.GPIO as GPIO

# Define GPIO pins for LEDs
LED_PINS = {
    "COVID": 17,
    "Symptomatic": 27,
    "Healthy": 22
}

def setup_leds():
    GPIO.setmode(GPIO.BCM)
    for pin in LED_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)

def reset_leds():
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)

def set_led_for_prediction(prediction):
    reset_leds()
    pin = LED_PINS.get(prediction)
    print('prediction:',prediction)
    print('pin:',pin)
    if pin:
        GPIO.output(pin, GPIO.LOW)


# Constants for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # Standard rate for librosa
CHUNK = 1024
RECORD_SECONDS = 15
TEMP_DIR = os.path.join(tempfile.gettempdir(), "audio_processor")

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

def load_model(model_path='cough_classification_model.pkl'):
    """Load the classification model from pickle file"""
    with open(model_path, 'rb') as f:
        components = pickle.load(f)
    return components

def extract_all_features(audio_path, sample_rate=None):
    """Extract comprehensive set of audio features"""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Basic features
    features = {}

    # Duration
    features['duration'] = librosa.get_duration(y=y, sr=sr)

    # RMS Energy
    features['rms_mean'] = np.mean(librosa.feature.rms(y=y)[0])
    features['rms_std'] = np.std(librosa.feature.rms(y=y)[0])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Spectral Features
    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_centroid_std'] = np.std(centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(bandwidth)
    features['spectral_bandwidth_std'] = np.std(bandwidth)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc{i + 1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i + 1}_std'] = np.std(mfccs[i])

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)

    return features

def process_audio_file(audio_file):
    """Process audio file and return prediction"""
    try:
        # Extract features
        features = extract_all_features(audio_file)

        # Load model and make prediction
        model_components = load_model()

        # Prepare features for prediction
        feature_names = model_components['feature_names']
        features_df = pd.DataFrame([features])
        features_df = features_df[feature_names]

        # Scale features
        features_scaled = model_components['scaler'].transform(features_df)

        # Predict
        prediction_idx = model_components['model'].predict(features_scaled)[0]
        prediction = model_components['label_encoder'].inverse_transform([prediction_idx])[0]

        # Get probabilities
        probs = model_components['model'].predict_proba(features_scaled)[0]
        class_probs = {
            model_components['label_encoder'].inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probs)
        }

        return prediction, class_probs
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None, None

def record_audio():
    """Record audio for RECORD_SECONDS and save to a temporary file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = os.path.join(TEMP_DIR, f"audio_snippet_{timestamp}.wav")

    p = pyaudio.PyAudio()

    try:
        print(f"Recording {RECORD_SECONDS} seconds of audio...")

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording complete.")

        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return temp_file
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None
    finally:
        p.terminate()

def cleanup_audio_files():
    """Clean up temporary audio files older than 5 minutes"""
    current_time = time.time()
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        # Check if the file is older than 5 minutes
        if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > 300:
            try:
                os.remove(file_path)
                print(f"Removed old file: {filename}")
            except Exception as e:
                print(f"Error removing file {filename}: {e}")

def cleanup_thread():
    """Background thread to periodically clean up old audio files"""
    while True:
        cleanup_audio_files()
        time.sleep(60)  # Check every minute

def main():
    """Main function to continuously record, process, and clean up audio"""
    print("Starting background audio processor...")
    print("Press Ctrl+C to stop.")

    # Start cleanup thread
    cleanup_thread_instance = threading.Thread(target=cleanup_thread, daemon=True)
    cleanup_thread_instance.start()
    setup_leds()

    try:
        while True:
            # Record audio
            audio_file = record_audio()

            if audio_file:
                # Process audio
                prediction, class_probs = process_audio_file(audio_file)

                if prediction:
                    # Print results
                    print("\n" + "="*50)
                    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Audio file: {os.path.basename(audio_file)}")
                    print(f"Prediction: {prediction}")
                    print("Probabilities:")
                    for cls, prob in class_probs.items():
                        print(f"  {cls}: {prob:.4f}")
                    print("="*50 + "\n")
                    
                    set_led_for_prediction(prediction)

            # Small delay before next recording
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping background audio processor...")
    finally:
        # Final cleanup
        cleanup_audio_files()
        GPIO.cleanup()
        # Try to remove the temp directory
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"Removed temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"Error removing temporary directory: {e}")

if __name__ == "__main__":
    main()