import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class CoughvidDataset:
    """Helper class to download and prepare the COUGHVID dataset for training"""
    
    def __init__(self, data_dir="coughvid_data"):
        """Initialize the dataset handler
        
        Args:
            data_dir: Directory to store the dataset
        """
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        self.metadata_path = os.path.join(data_dir, "metadata_compiled.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def download_dataset(self, limit=None):
        """Download the COUGHVID dataset using kagglehub
        
        Args:
            limit: Maximum number of files to use (for testing)
        
        Returns:
            Path to the directory containing the audio files and metadata
        """
        try:
            import kagglehub
        except ImportError:
            raise ImportError("Please install kagglehub: pip install kagglehub")
            
        print("Downloading COUGHVID dataset using kagglehub...")
        # Download latest version directly
        kaggle_path = kagglehub.dataset_download("nasrulhakim86/coughvid-wav")
        print(f"Dataset downloaded to: {kaggle_path}")
        
        try:
            # Look for metadata in the Kaggle download directory
            metadata_path = None
            for root, _, files in os.walk(kaggle_path):
                for file in files:
                    if file.endswith('.csv') and 'metadata' in file.lower():
                        metadata_path = os.path.join(root, file)
                        break
            
            if not metadata_path:
                # If no metadata file found, use the original path
                if os.path.exists(self.metadata_path):
                    metadata_path = self.metadata_path
                else:
                    raise FileNotFoundError("Could not find metadata file in the downloaded dataset.")
            
            # Load metadata
            metadata = pd.read_csv(
                metadata_path,
                quoting=pd.io.common.csv.QUOTE_NONE,
                escapechar='\\',
                on_bad_lines='skip'
            )
            
            # Filter to get records with COVID-19 status
            valid_records = metadata[
                (metadata['status'] == 'COVID-19') | 
                (metadata['status'] == 'healthy')
            ].copy()
            
            # Apply limit if specified
            if limit:
                valid_records = valid_records.head(limit)
                
            return kaggle_path, valid_records
            
        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise
    
    def prepare_data_for_training(self, limit=None, convert_to_wav=False):
        """Prepare the dataset for training
        
        Args:
            limit: Maximum number of files to use (for testing)
            convert_to_wav: Whether to convert audio files to WAV format (not needed for kaggle dataset)
            
        Returns:
            Lists of healthy and unhealthy audio files
        """
        # Download dataset using kagglehub
        dataset_path, metadata = self.download_dataset(limit=limit)
        
        # Create subdirectories for healthy and sick
        healthy_dir = os.path.join(self.data_dir, "healthy")
        sick_dir = os.path.join(self.data_dir, "sick")
        os.makedirs(healthy_dir, exist_ok=True)
        os.makedirs(sick_dir, exist_ok=True)
        
        # Process files
        healthy_files = []
        sick_files = []
        
        # Search for wav files in the downloaded dataset
        wav_files = {}
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_files[file.split('.')[0]] = os.path.join(root, file)
        
        for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
            # Source file path
            uuid = str(row['uuid'])
            
            if uuid in wav_files:
                wav_path = wav_files[uuid]
                
                # Determine destination based on status
                if row['status'] == 'healthy':
                    dest_dir = healthy_dir
                    output_list = healthy_files
                else:  # COVID-19
                    dest_dir = sick_dir
                    output_list = sick_files
                
                # Copy or link to the destination if needed
                dest_path = os.path.join(dest_dir, f"{uuid}.wav")
                if not os.path.exists(dest_path):
                    # Option 1: Create a symbolic link (efficient but platform-dependent)
                    # os.symlink(wav_path, dest_path)
                    
                    # Option 2: Copy the file (works everywhere but uses more disk space)
                    import shutil
                    shutil.copy2(wav_path, dest_path)
                
                output_list.append(dest_path)
        
        print(f"Prepared {len(healthy_files)} healthy and {len(sick_files)} sick audio files")
        return healthy_files, sick_files

if __name__ == "__main__":
    # Simple test
    dataset = CoughvidDataset()
    healthy_files, sick_files = dataset.prepare_data_for_training(limit=100)
    print(f"Healthy files: {len(healthy_files)}")
    print(f"Sick files: {len(sick_files)}")

