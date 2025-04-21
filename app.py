import pickle
import numpy as np
import pandas as pd
import librosa
import gradio as gr
import soundfile as sf


def load_model(model_path='cough_classification_model.pkl'):
    with open(model_path, 'rb') as f:
        components = pickle.load(f)

    return components


# Extract features from audio
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
    """Process uploaded audio file and return features and prediction"""
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

    # Format the outputs
    feature_output = "Extracted Features:\n"
    for feat_name, feat_value in features.items():
        feature_output += f"{feat_name}: {feat_value:.4f}\n"

    prediction_output = f"\nPrediction: {prediction}\n\nProbabilities:\n"
    for cls, prob in class_probs.items():
        prediction_output += f"{cls}: {prob:.4f}\n"

    return feature_output, prediction_output

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Cough Feature Extractor and Analyzer") as demo:
        gr.Markdown("# Cough Feature Extractor and Analyzer")
        gr.Markdown("Upload an audio file containing a cough to extract features and analyze its health status.")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")

        with gr.Row():
            feature_output = gr.Textbox(label="Extracted Features", lines=20)
            prediction_output = gr.Textbox(label="Prediction Results", lines=10)

        analyze_btn = gr.Button("Analyze Audio")
        analyze_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[feature_output, prediction_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
