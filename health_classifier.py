import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from transformers import ViTConfig, ViTModel
import importlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HealthClassifier:
    """
    A classifier to determine if someone is sick based on HeAR embeddings
    """
    def __init__(self, model_path=None):
        """
        Initialize the health classifier
        
        Args:
            model_path: Path to a saved classifier model (if available)
        """
        if model_path and os.path.exists(model_path):
            self.classifier = joblib.load(model_path)
            self.is_trained = True
        else:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.is_trained = False
            
        # Load HeAR model for generating embeddings
        self.hear_model = self._load_hear_model()
        
        # Import audio utilities
        try:
            audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
            self.preprocess_audio = audio_utils.preprocess_audio
            self.resample_audio = audio_utils.resample_audio_and_convert_to_mono
        except ImportError:
            print("Warning: Cannot import audio utilities from HeAR. Make sure the repo is cloned.")
            
    def _load_hear_model(self):
        """Load the HeAR model from Hugging Face Hub"""
        configuration = ViTConfig(
            image_size=(192, 128),
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=1024 * 4,
            hidden_act="gelu_fast",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            pooled_dim=512,
            patch_size=16,
            num_channels=1,
            qkv_bias=True,
            encoder_stride=16,
            pooler_act='linear',
            pooler_output_size=512,
        )
        
        try:
            model = ViTModel.from_pretrained(
                "google/hear-pytorch",
                config=configuration
            )
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading HeAR model: {e}")
            return None
    
    def generate_embedding(self, audio_array, sample_rate=16000):
        """
        Generate HeAR embedding from audio array
        
        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Embedding vector as numpy array
        """
        CLIP_DURATION = 2
        CLIP_LENGTH = sample_rate * CLIP_DURATION
        
        # Ensure audio is at 16kHz sample rate
        if sample_rate != 16000:
            audio_array = self.resample_audio(
                audio_array=audio_array,
                sampling_rate=sample_rate,
                new_sampling_rate=16000
            )
            sample_rate = 16000
            
        # Ensure audio is at least 2 seconds
        if len(audio_array) < CLIP_LENGTH:
            # Pad with zeros if necessary
            audio_array = np.pad(audio_array, (0, CLIP_LENGTH - len(audio_array)), 'constant')
        
        # Take the first 2 seconds
        clip = audio_array[:CLIP_LENGTH]
        
        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(np.expand_dims(clip, axis=0)).float()
        
        # Generate embedding
        with torch.no_grad():
            output = self.hear_model(
                self.preprocess_audio(input_tensor), 
                return_dict=True
            )
            
        # Extract embedding vector
        embedding = output.pooler_output.cpu().numpy().flatten()
        return embedding
    
    def train(self, healthy_audio_files, unhealthy_audio_files, save_path=None):
        """
        Train the classifier using healthy and unhealthy audio files
        
        Args:
            healthy_audio_files: List of paths to audio files from healthy people
            unhealthy_audio_files: List of paths to audio files from sick people
            save_path: Path to save the trained model
            
        Returns:
            Accuracy score of the trained model
        """
        from scipy.io import wavfile
        
        # Process healthy audio files
        healthy_embeddings = []
        for file_path in healthy_audio_files:
            try:
                sample_rate, audio = wavfile.read(file_path)
                embedding = self.generate_embedding(audio, sample_rate)
                healthy_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        # Process unhealthy audio files
        unhealthy_embeddings = []
        for file_path in unhealthy_audio_files:
            try:
                sample_rate, audio = wavfile.read(file_path)
                embedding = self.generate_embedding(audio, sample_rate)
                unhealthy_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Create dataset
        X = np.vstack([healthy_embeddings, unhealthy_embeddings])
        y = np.array([0] * len(healthy_embeddings) + [1] * len(unhealthy_embeddings))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Classifier accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Sick']))
        
        # Save model if specified
        if save_path:
            joblib.dump(self.classifier, save_path)
            print(f"Model saved to {save_path}")
            
        return accuracy
    
    def predict(self, audio_array, sample_rate=16000):
        """
        Predict if a person is sick based on audio
        
        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with prediction (0=healthy, 1=sick) and probability
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained yet. Call train() first.")
            
        # Generate embedding
        embedding = self.generate_embedding(audio_array, sample_rate)
        
        # Make prediction
        prediction = self.classifier.predict([embedding])[0]
        probability = self.classifier.predict_proba([embedding])[0]
        
        return {
            "prediction": int(prediction),
            "status": "Sick" if prediction == 1 else "Healthy",
            "confidence": float(probability[prediction])
        }
