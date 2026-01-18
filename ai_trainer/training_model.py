"""
Training Pipeline for Sign Language Recognition
Object-oriented implementation for data preparation and model training
"""
import os
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# ======================================================
# PATH CONFIGURATION FOR IMPORTING CONFIG
# ======================================================
# Get current script directory (ai_trainer)
script_dir = Path(__file__).resolve().parent
print(f"📂 Script directory: {script_dir}")

# Go up one level to reach project root directory (new)
project_root = script_dir.parent
print(f"📂 Project root directory: {project_root}")

# Add root directory to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify that config.py exists
config_path = project_root / "config.py"
print(f"📂 Looking for config at: {config_path}")
print(f"📂 Does config.py exist?: {config_path.exists()}")

# Import shared configuration
try:
    from config import (
        get_model_paths, get_latest_video_dir, get_video_files,
        load_processing_state, save_processing_state,
        ensure_directories, validate_environment,
        PREDICTIONS_OUTPUT, PREDICTION_THRESHOLD, MODEL_FRAMES,
        VALIDATED_VIDEOS_DIR, VALIDATED_KEYPOINTS_DIR,
        UNKNOWN_VIDEOS_DIR, UNKNOWN_KEYPOINTS_DIR,
        get_whisper_and_video_paths, TRAIN_DATA_DIR,
        WORDS_JSON_PATH, KEYPOINTS_PATH, MODEL_PATH,
        LENGTH_KEYPOINTS, MODEL_FILE, WORDS_JSON_FILE,
        get_current_model_dir, get_latest_model_dir, get_model_files_from_dir
    )
    CONFIG_AVAILABLE = True
    print("✓ Shared config loaded successfully")
except ImportError as e:
    print(f"⚠ Could not import config.py: {e}")
    print(f"⚠ Using default configuration")
    CONFIG_AVAILABLE = False
    # Default values if config is not available
    TRAIN_DATA_DIR = Path("./train_data")
    PREDICTIONS_OUTPUT = Path("./output")
    PREDICTION_THRESHOLD = 0.7
    MODEL_FRAMES = 30
    LENGTH_KEYPOINTS = 1662
    MODELS_DIR = Path("./models")
    KEYPOINTS_PATH = MODELS_DIR / "keypoints"
    MODEL_PATH = MODELS_DIR / "actions_15.keras"
    WORDS_JSON_PATH = MODELS_DIR / "words.json"
    VALIDATED_VIDEOS_DIR = TRAIN_DATA_DIR / "validadas" / "videos"
    VALIDATED_KEYPOINTS_DIR = TRAIN_DATA_DIR / "validadas" / "keypoints"
    UNKNOWN_VIDEOS_DIR = TRAIN_DATA_DIR / "desconocidas" / "videos"
    UNKNOWN_KEYPOINTS_DIR = TRAIN_DATA_DIR / "desconocidas" / "keypoints"
    
    def ensure_directories():
        """Creates necessary directories if they don't exist."""
        PREDICTIONS_OUTPUT.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        KEYPOINTS_PATH.mkdir(parents=True, exist_ok=True)
        VALIDATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATED_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        UNKNOWN_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        UNKNOWN_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        print("✓ Directories verified/created")


# ======================================================
# DATA PREPARATION CLASS
# ======================================================
class DataPreparation:
    """
    Handles data preparation tasks including word discovery,
    keypoints loading, and HDF5 file creation.
    """
    
    def __init__(self, train_data_path: Optional[Path] = None, 
                 model_dir: Optional[Path] = None):
        """
        Initialize DataPreparation with train data path and model directory.
        
        Args:
            train_data_path: Path to train_data directory (uses TRAIN_DATA_DIR if None)
            model_dir: Path to model directory where words.json will be saved
        """
        self.train_data_path = train_data_path or TRAIN_DATA_DIR
        self.model_dir = model_dir
        self.words_json_path = None
        self.keypoints_output_path = KEYPOINTS_PATH
        
        # Set words_json_path based on model_dir
        if self.model_dir:
            self.words_json_path = self.model_dir / WORDS_JSON_FILE
        else:
            self.words_json_path = WORDS_JSON_PATH
        
    def generate_words_json(self) -> List[str]:
        """
        Generates words.json file based on folders found in
        train_data/validadas/keypoints/ and train_data/desconocidas/keypoints/
        
        Returns:
            List of word_ids found
        """
        word_ids = []
        
        # Search in validated/keypoints
        validated_keypoints_path = self.train_data_path / "validadas" / "keypoints"
        
        if validated_keypoints_path.exists():
            folders = [f.name for f in validated_keypoints_path.iterdir() if f.is_dir()]
            word_ids.extend(folders)
        else:
            print(f"⚠ Folder not found: {validated_keypoints_path}")
        
        # Also include words from unknown/keypoints if they exist
        unknown_keypoints_path = self.train_data_path / "desconocidas" / "keypoints"
        if unknown_keypoints_path.exists():
            folders = [f.name for f in unknown_keypoints_path.iterdir() 
                      if f.is_dir() and f.name != "sin_palabra"]
            word_ids.extend(folders)
        
        # Remove duplicates and sort
        word_ids = sorted(list(set(word_ids)))
        
        if len(word_ids) == 0:
            print("⚠ No words found. Check folder structure.")
            return []
        
        # Create JSON structure
        words_data = {"word_ids": word_ids}
        
        # Create models folder if it doesn't exist
        self.words_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(self.words_json_path, 'w', encoding='utf-8') as f:
            json.dump(words_data, f, indent=4, ensure_ascii=False)
        
        print(f"✓ words.json generated at: {self.words_json_path}")
        print(f"✓ Total words found: {len(word_ids)}")
        for word in word_ids:
            print(f"  - {word}")
        
        return word_ids
    
    @staticmethod
    def load_keypoints_sequence_from_json(json_path: Path) -> np.ndarray:
        """
        Loads ONE COMPLETE SEQUENCE of keypoints from a JSON file.
        
        Args:
            json_path: Path to JSON file with keypoints
        
        Returns:
            numpy.array: Array with keypoints sequence (shape: (num_frames, 1662))
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON should be a list of frames
            # Each frame is a list of 1662 keypoints
            # Format: [[frame1_kp1, frame1_kp2, ...], [frame2_kp1, frame2_kp2, ...], ...]
            
            if isinstance(data, list):
                kp_sequence = np.array(data, dtype=np.float32)
                
                # Verify dimensions
                if len(kp_sequence.shape) == 2:
                    num_frames, num_keypoints = kp_sequence.shape
                    
                    if num_keypoints == LENGTH_KEYPOINTS:
                        return kp_sequence
                    else:
                        print(f"⚠ Incorrect keypoints per frame in {json_path}: "
                              f"{num_keypoints} (expected: {LENGTH_KEYPOINTS})")
                        # Try to correct or pad
                        if num_keypoints < LENGTH_KEYPOINTS:
                            # Pad with zeros
                            padding = np.zeros((num_frames, LENGTH_KEYPOINTS - num_keypoints), 
                                             dtype=np.float32)
                            kp_sequence = np.concatenate([kp_sequence, padding], axis=1)
                        else:
                            # Truncate
                            kp_sequence = kp_sequence[:, :LENGTH_KEYPOINTS]
                        return kp_sequence
                else:
                    print(f"⚠ Incorrect dimensions in {json_path}: {kp_sequence.shape}")
                    return np.zeros((1, LENGTH_KEYPOINTS), dtype=np.float32)
            
            # If it's a dictionary, look for "keypoints" or similar key
            if isinstance(data, dict):
                if "keypoints" in data:
                    kp_sequence = np.array(data["keypoints"], dtype=np.float32)
                    if len(kp_sequence.shape) == 2 and kp_sequence.shape[1] == LENGTH_KEYPOINTS:
                        return kp_sequence
                
                if "frames" in data:
                    kp_sequence = np.array(data["frames"], dtype=np.float32)
                    if len(kp_sequence.shape) == 2 and kp_sequence.shape[1] == LENGTH_KEYPOINTS:
                        return kp_sequence
            
            print(f"⚠ Unrecognized JSON format in {json_path}")
            return np.zeros((1, LENGTH_KEYPOINTS), dtype=np.float32)
            
        except Exception as e:
            print(f"⚠ Error loading {json_path}: {e}")
            return np.zeros((1, LENGTH_KEYPOINTS), dtype=np.float32)
    
    def create_keypoints_h5_from_json(self, word_id: str) -> None:
        """
        Creates HDF5 file with keypoints for a word from JSON files.
        Each JSON contains a COMPLETE SEQUENCE (multiple frames).
        
        Args:
            word_id: Word identifier (folder name)
        """
        output_hdf_path = self.keypoints_output_path / f"{word_id}.h5"
        data = pd.DataFrame([])
        
        # Search for keypoints in validated and unknown
        keypoints_paths = [
            self.train_data_path / "validadas" / "keypoints" / word_id,
            self.train_data_path / "desconocidas" / "keypoints" / word_id
        ]
        
        sample_number = 0
        total_samples = 0
        total_json_files = 0
        
        print(f'Creating keypoints for "{word_id}"...')
        
        for keypoints_folder in keypoints_paths:
            if not keypoints_folder.exists():
                continue
            
            # Get ALL JSON files from the folder
            json_files = sorted([f for f in keypoints_folder.iterdir() if f.suffix == '.json'])
            
            total_json_files += len(json_files)
            
            print(f"  📁 Folder: {keypoints_folder}")
            print(f"  📄 JSON files found: {len(json_files)}")
            if len(json_files) > 0:
                file_names = [f.name for f in json_files[:5]]
                print(f"     Files: {', '.join(file_names)}{' ...' if len(json_files) > 5 else ''}")
            
            # Process EACH JSON file (each one is a different sample)
            for json_file in json_files:
                sample_number += 1
                
                # Load COMPLETE SEQUENCE of keypoints from JSON file
                keypoints_sequence = self.load_keypoints_sequence_from_json(json_file)
                
                # keypoints_sequence has shape (num_frames, 1662)
                # We need to add each frame to the DataFrame
                num_frames = len(keypoints_sequence)
                
                for frame_idx, frame_keypoints in enumerate(keypoints_sequence):
                    frame_data = {
                        'sample': sample_number,
                        'frame': frame_idx + 1,
                        'keypoints': [frame_keypoints]
                    }
                    df_keypoints = pd.DataFrame(frame_data)
                    data = pd.concat([data, df_keypoints], ignore_index=True)
                
                total_samples += 1
                print(f"  ✓ Sample {total_samples}: {json_file.name} ({num_frames} frames)", 
                      end="\r")
        
        print()  # New line after \r
        
        if total_samples == 0:
            print(f"⚠ No JSON files found for '{word_id}'")
            print(f"   Verify that the folder with .json files exists")
            return
        
        # Create keypoints folder if it doesn't exist
        output_hdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to HDF5
        data.to_hdf(output_hdf_path, key="data", mode="w")
        print(f"✓ '{word_id}': {total_samples} samples ({total_json_files} JSON files) "
              f"→ {output_hdf_path}")
    
    def process_all_words(self) -> List[str]:
        """
        Processes all words: generates words.json and creates HDF5 files for each word.
        
        Returns:
            List of processed word_ids
        """
        # 1. Generate words.json
        print("\n" + "="*60)
        print("STEP 1: Generating words.json")
        print("="*60)
        word_ids = self.generate_words_json()
        
        if len(word_ids) == 0:
            print("⚠ No words to process. Terminating.")
            return []
        
        # 2. Create HDF5 files for each word
        print("\n" + "="*60)
        print("STEP 2: Creating HDF5 keypoints files")
        print("="*60)
        
        for i, word_id in enumerate(word_ids, 1):
            print(f"\n[{i}/{len(word_ids)}] Processing: {word_id}")
            self.create_keypoints_h5_from_json(word_id)
        
        print("\n" + "="*60)
        print("✓ PROCESS COMPLETED")
        print("="*60)
        print(f"✓ words.json generated: {self.words_json_path}")
        print(f"✓ {len(word_ids)} HDF5 files created in: {self.keypoints_output_path}")
        
        return word_ids


# ======================================================
# DATA LOADER CLASS
# ======================================================
class DataLoader:
    """
    Handles loading of training data from HDF5 files and word IDs.
    """
    
    def __init__(self, words_json_path: Optional[Path] = None, 
                 keypoints_path: Optional[Path] = None):
        """
        Initialize DataLoader with paths.
        
        Args:
            words_json_path: Path to words.json file
            keypoints_path: Path to keypoints HDF5 directory
        """
        self.words_json_path = words_json_path or WORDS_JSON_PATH
        self.keypoints_path = keypoints_path or KEYPOINTS_PATH
    
    def get_word_ids(self) -> List[str]:
        """
        Loads word_ids from words.json file.
        
        Returns:
            List of word identifiers
        """
        try:
            with open(self.words_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('word_ids', [])
        except Exception as e:
            print(f"⚠ Error loading words.json: {e}")
            return []
    
    def load_sequences_for_word(self, word_id: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Loads all sequences for a specific word from its HDF5 file.
        
        Args:
            word_id: Word identifier
        
        Returns:
            Tuple of (sequences_list, labels_list)
        """
        hdf_path = self.keypoints_path / f"{word_id}.h5"
        
        if not hdf_path.exists():
            print(f"⚠ HDF5 file not found for word '{word_id}': {hdf_path}")
            return [], []
        
        try:
            df = pd.read_hdf(hdf_path, key='data')
            
            sequences = []
            samples = df['sample'].unique()
            
            for sample_id in samples:
                sample_frames = df[df['sample'] == sample_id].sort_values('frame')
                keypoints = np.vstack(sample_frames['keypoints'].values)
                sequences.append(keypoints)
            
            return sequences
            
        except Exception as e:
            print(f"⚠ Error loading HDF5 for '{word_id}': {e}")
            return []
    
    def get_sequences_and_labels(self, word_ids: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Loads all sequences and their corresponding labels.
        
        Args:
            word_ids: List of word identifiers
        
        Returns:
            Tuple of (all_sequences, all_labels)
        """
        all_sequences = []
        all_labels = []
        
        for label, word_id in enumerate(word_ids):
            sequences = self.load_sequences_for_word(word_id)
            
            if len(sequences) > 0:
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
                print(f"  ✓ {word_id}: {len(sequences)} sequences loaded")
            else:
                print(f"  ⚠ {word_id}: No sequences found")
        
        return all_sequences, all_labels
    
    @staticmethod
    def normalize_keypoints(sequence: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Normalizes a keypoints sequence to a specific number of frames.
        
        Args:
            sequence: Input sequence (num_frames, num_keypoints)
            target_frames: Target number of frames
        
        Returns:
            Normalized sequence (target_frames, num_keypoints)
        """
        current_frames = len(sequence)
        
        if current_frames == target_frames:
            return sequence
        elif current_frames > target_frames:
            # Downsample
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            return sequence[indices]
        else:
            # Upsample by repeating frames
            repeat_factor = target_frames // current_frames
            remainder = target_frames % current_frames
            
            repeated = np.repeat(sequence, repeat_factor, axis=0)
            
            if remainder > 0:
                extra_indices = np.linspace(0, current_frames - 1, remainder, dtype=int)
                extra_frames = sequence[extra_indices]
                repeated = np.concatenate([repeated, extra_frames], axis=0)
            
            return repeated[:target_frames]


# ======================================================
# MODEL BUILDER CLASS
# ======================================================
class ModelBuilder:
    """
    Handles model architecture creation and configuration.
    """
    
    @staticmethod
    def build_lstm_model(frames: int, num_classes: int, num_keypoints: int = LENGTH_KEYPOINTS) -> Sequential:
        """
        Creates LSTM model for sign language recognition.
        
        Args:
            frames: Number of frames per sequence
            num_classes: Number of classes/words to recognize
            num_keypoints: Number of keypoints per frame
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(128, return_sequences=True, activation='relu', 
                       input_shape=(frames, num_keypoints)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        
        # Third LSTM layer
        model.add(LSTM(128, return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


# ======================================================
# MODEL TRAINER CLASS
# ======================================================
class ModelTrainer:
    """
    Handles model training process including data loading,
    normalization, and training execution.
    """
    
    def __init__(self, model_dir: Optional[Path] = None, 
                 keypoints_path: Optional[Path] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            model_dir: Directory where model and words.json will be saved
            keypoints_path: Path to keypoints directory
        """
        self.model_dir = model_dir
        
        # Set paths based on model_dir
        if self.model_dir:
            self.model_path = self.model_dir / MODEL_FILE
            self.words_json_path = self.model_dir / WORDS_JSON_FILE
        else:
            self.model_path = MODEL_PATH
            self.words_json_path = WORDS_JSON_PATH
            
        self.data_loader = DataLoader(self.words_json_path, keypoints_path)
        self.model_builder = ModelBuilder()
        self.model = None
        self.history = None
    
    def train(self, epochs: int = 500, batch_size: int = 8, 
              validation_split: float = 0.15, target_frames: int = MODEL_FRAMES) -> Tuple[Sequential, Dict]:
        """
        Trains the sign language recognition model.
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size
            validation_split: Percentage of data for validation
            target_frames: Number of frames to normalize sequences to
        
        Returns:
            Tuple of (trained_model, training_history)
        """
        print("\n" + "="*60)
        print("SIGN LANGUAGE RECOGNITION MODEL TRAINING")
        print("="*60)
        
        # 1. Load word_ids from JSON
        print("\n[1/6] Loading words from words.json...")
        word_ids = self.data_loader.get_word_ids()
        print(f"✓ {len(word_ids)} words loaded: {word_ids}")
        
        # 2. Load sequences and labels from HDF5 files
        print("\n[2/6] Loading keypoints sequences from HDF5 files...")
        sequences, labels = self.data_loader.get_sequences_and_labels(word_ids)
        print(f"✓ {len(sequences)} sequences loaded")
        
        if len(sequences) == 0:
            print("⚠ ERROR: No sequences found. Check HDF5 files.")
            return None, None
        
        # 3. Normalize sequences to target_frames
        print(f"\n[3/6] Normalizing sequences to {target_frames} frames...")
        normalized_sequences = []
        for i, seq in enumerate(sequences):
            normalized_seq = self.data_loader.normalize_keypoints(seq, target_frames)
            normalized_sequences.append(normalized_seq)
            if (i + 1) % 50 == 0:
                print(f"  Normalized {i + 1}/{len(sequences)} sequences", end="\r")
        
        print(f"\n✓ All sequences normalized to shape ({target_frames}, {LENGTH_KEYPOINTS})")
        
        # 4. Prepare training data
        print("\n[4/6] Preparing data for training...")
        X = np.array(normalized_sequences, dtype=np.float32)
        y = to_categorical(labels, num_classes=len(word_ids)).astype(int)
        
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        print(f"✓ Number of classes: {len(word_ids)}")
        
        # 5. Split into training and validation
        print(f"\n[5/6] Splitting data (train: {int((1-validation_split)*100)}%, "
              f"val: {int(validation_split*100)}%)...")
        
        # Check if there are enough samples to split with stratify
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples_per_class = np.min(counts)
        
        if min_samples_per_class < 2:
            print(f"⚠ WARNING: Some classes have only {min_samples_per_class} sample(s)")
            print("  Cannot use stratification. Splitting without stratify...")
            print("  RECOMMENDATION: Add more samples (videos) per word for better training")
            
            if len(sequences) < 10:
                print(f"\n⚠ Only {len(sequences)} total samples. Training without validation.")
                X_train = X
                y_train = y
                X_val = X[:2] if len(X) >= 2 else X
                y_val = y[:2] if len(y) >= 2 else y
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=labels
            )
        
        print(f"✓ Training data: {X_train.shape[0]} samples")
        print(f"✓ Validation data: {X_val.shape[0]} samples")
        
        # 6. Create and train model
        print(f"\n[6/6] Creating and training model...")
        print("-" * 60)
        
        # Create model
        self.model = self.model_builder.build_lstm_model(target_frames, len(word_ids))
        
        # Show model summary
        print("\nModel architecture:")
        self.model.summary()
        print("-" * 60)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create models folder if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            str(self.model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        print(f"\nStarting training (maximum {epochs} epochs)...")
        print("="*60)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # 7. Final results
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        # Evaluate on validation set
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n✓ Model saved at: {self.model_path}")
        print(f"✓ Epochs trained: {len(self.history.history['loss'])}")
        print(f"✓ Final validation accuracy: {val_accuracy*100:.2f}%")
        print(f"✓ Final validation loss: {val_loss:.4f}")
        
        # Show class distribution
        print("\n" + "="*60)
        print("SAMPLE DISTRIBUTION PER CLASS")
        print("="*60)
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            word = word_ids[label_idx]
            print(f"  {word:20s}: {count:3d} samples")
        
        print("\n" + "="*60)
        
        return self.model, self.history


# ======================================================
# MAIN TRAINING PIPELINE CLASS
# ======================================================
class TrainingPipeline:
    """
    Main class that orchestrates the complete training pipeline.
    """
    
    def __init__(self, train_data_path: Optional[Path] = None,
                 create_timestamped_dir: bool = True):
        """
        Initialize the complete training pipeline.
        
        Args:
            train_data_path: Path to train_data directory
            create_timestamped_dir: Whether to create timestamped model directory
        """
        # Create timestamped model directory if requested
        if create_timestamped_dir and CONFIG_AVAILABLE:
            self.model_dir = get_current_model_dir()
            print(f"\n📁 Created timestamped model directory: {self.model_dir.name}")
        else:
            self.model_dir = None
            print(f"\n📁 Using default model directory")
        
        self.data_preparation = DataPreparation(train_data_path, self.model_dir)
        self.model_trainer = ModelTrainer(self.model_dir)
    
    def run_full_pipeline(self, epochs: int = 500, batch_size: int = 8,
                         validation_split: float = 0.15, 
                         prepare_data: bool = True) -> Tuple[Sequential, Dict]:
        """
        Runs the complete training pipeline from data preparation to model training.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Validation split percentage
            prepare_data: Whether to run data preparation step
        
        Returns:
            Tuple of (trained_model, training_history)
        """
        # Step 1: Data Preparation (optional)
        if prepare_data:
            print("\n" + "="*70)
            print("PHASE 1: DATA PREPARATION")
            print("="*70)
            word_ids = self.data_preparation.process_all_words()
            
            if len(word_ids) == 0:
                print("⚠ No data to train on. Pipeline terminated.")
                return None, None
        
        # Step 2: Model Training
        print("\n" + "="*70)
        print("PHASE 2: MODEL TRAINING")
        print("="*70)
        model, history = self.model_trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        # Print final location of saved files
        if model is not None and self.model_dir:
            print("\n" + "="*70)
            print("📦 SAVED FILES LOCATION")
            print("="*70)
            print(f"Model directory: {self.model_dir}")
            print(f"  • Model file: {self.model_trainer.model_path.name}")
            print(f"  • Words JSON: {self.model_trainer.words_json_path.name}")
            print(f"\nFull paths:")
            print(f"  • {self.model_trainer.model_path}")
            print(f"  • {self.model_trainer.words_json_path}")
        
        return model, history


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    
    print("="*70)
    print("SIGN LANGUAGE RECOGNITION - TRAINING PIPELINE")
    print("="*70)
    print("\nConfiguration:")
    print(f"  - Frames per sequence: {MODEL_FRAMES}")
    print(f"  - Keypoints per frame: {LENGTH_KEYPOINTS}")
    print(f"  - Train data path: {TRAIN_DATA_DIR}")
    print(f"  - Keypoints HDF5 path: {KEYPOINTS_PATH}")
    
    # Create pipeline instance with timestamped directory
    pipeline = TrainingPipeline(create_timestamped_dir=True)
    
    # Run complete pipeline
    model, history = pipeline.run_full_pipeline(
        epochs=500,
        batch_size=8,
        validation_split=0.15,
        prepare_data=True  # Set to False if data is already prepared
    )
    
    if model is not None:
        print("\n✓ Pipeline completed successfully!")
        if pipeline.model_dir:
            print(f"\nModel saved in timestamped directory:")
            print(f"  {pipeline.model_dir}")
    else:
        print("\n⚠ Pipeline terminated with errors.")