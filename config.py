"""
Shared configuration file between data_miner and predictor.
This file must be accessible to both environments.
"""
import os
import json
from pathlib import Path
from datetime import datetime


# ======================================================
# PROJECT BASE DIRECTORY
# ======================================================
# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


# ======================================================
# DIRECTORY PATHS
# ======================================================
# Directory where data_miner saves processed videos
DATA_MINER_OUTPUT = PROJECT_ROOT / "vid_data"

# Main training directory
TRAIN_DATA_DIR = PROJECT_ROOT / "train_data"

# Validated folders structure
VALIDATED_DIR = TRAIN_DATA_DIR / "validadas"
VALIDATED_VIDEOS_DIR = VALIDATED_DIR / "videos"
VALIDATED_KEYPOINTS_DIR = VALIDATED_DIR / "keypoints"

# Unknown folders structure
UNKNOWN_DIR = TRAIN_DATA_DIR / "desconocidas"
UNKNOWN_VIDEOS_DIR = UNKNOWN_DIR / "videos"
UNKNOWN_KEYPOINTS_DIR = UNKNOWN_DIR / "keypoints"

# Maintain compatibility with existing code (deprecated)
VALIDATED_CLIPS_DIR = VALIDATED_VIDEOS_DIR
UNKNOWN_CLIPS_DIR = UNKNOWN_VIDEOS_DIR

# Directory where trained models are stored
MODELS_DIR = PROJECT_ROOT / "models"

# Directory for keypoints HDF5 files
KEYPOINTS_PATH = MODELS_DIR / "keypoints"

# Output directory for predictions
PREDICTIONS_OUTPUT = PROJECT_ROOT / "output"


# ======================================================
# SHARED STATE FILE
# ======================================================
# This JSON file will contain information about the last processing
STATE_FILE = PROJECT_ROOT / "last_processing_state.json"


# ======================================================
# MODEL CONFIGURATION
# ======================================================
# Model file names (deprecated - use get_current_model_dir())
MODEL_FILE = "actions_15.keras"
MODEL_PATH = MODELS_DIR / MODEL_FILE
WORDS_JSON_FILE = "words.json"
WORDS_JSON_PATH = MODELS_DIR / WORDS_JSON_FILE


def get_current_model_dir():
    """
    Creates and returns a timestamped model directory.
    Format: models/models_YYYYMMDD_HHMMSS
    
    Returns:
        Path: Path to the timestamped model directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"models_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_latest_model_dir():
    """
    Gets the most recent model directory.
    
    Returns:
        Path: Path to the latest model directory or None if no models exist
    """
    if not MODELS_DIR.exists():
        return None
    
    # Find all model directories
    model_dirs = [d for d in MODELS_DIR.iterdir() 
                  if d.is_dir() and d.name.startswith('models_')]
    
    if not model_dirs:
        return None
    
    # Sort by name (timestamp) and return the latest
    latest = sorted(model_dirs, key=lambda x: x.name)[-1]
    return latest


def get_model_files_from_dir(model_dir):
    """
    Gets paths to model files within a specific model directory.
    
    Args:
        model_dir: Path to model directory
    
    Returns:
        dict: Dictionary with paths to model and words.json
    """
    if not model_dir or not model_dir.exists():
        return None
    
    return {
        'model': model_dir / MODEL_FILE,
        'words_json': model_dir / WORDS_JSON_FILE,
        'dir': model_dir
    }


# ======================================================
# PROCESSING CONFIGURATION
# ======================================================
# Confidence threshold for predictions (0.0 - 1.0)
PREDICTION_THRESHOLD = 0.7

# Seconds between each segment in data_miner
SEGMENT_DURATION = 2

# Frames expected by the model
MODEL_FRAMES = 30  # Adjust according to your model

# Number of keypoints per frame (MediaPipe Holistic)
LENGTH_KEYPOINTS = 1662


# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def ensure_directories():
    """Creates necessary directories if they don't exist."""
    DATA_MINER_OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    KEYPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Create train_data structure
    VALIDATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATED_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    UNKNOWN_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    UNKNOWN_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories verified/created")


def get_latest_video_dir():
    """
    Gets the directory of the most recently processed video.
    
    Returns:
        Path: Path to the most recent video directory or None
    """
    if not DATA_MINER_OUTPUT.exists():
        return None
    
    video_dirs = [d for d in DATA_MINER_OUTPUT.iterdir() 
                  if d.is_dir() and d.name.startswith('video_')]
    
    if not video_dirs:
        return None
    
    # Sort by name (which includes timestamp)
    latest = sorted(video_dirs, key=lambda x: x.name)[-1]
    return latest


def get_model_paths():
    """
    Gets complete paths to model files from the latest model directory.
    
    Returns:
        tuple: (model_path, words_json_path) from latest model, or default paths
    """
    latest_dir = get_latest_model_dir()
    
    if latest_dir:
        model_path = latest_dir / MODEL_FILE
        words_path = latest_dir / WORDS_JSON_FILE
    else:
        # Fallback to root model directory if no timestamped dirs exist
        model_path = MODEL_PATH
        words_path = WORDS_JSON_PATH
    
    return model_path, words_path


def get_video_files(video_dir):
    """
    Gets paths to processed video files.
    
    Args:
        video_dir: Path to video directory
    
    Returns:
        dict: Dictionary with file paths
    """
    if not video_dir or not video_dir.exists():
        return None
    
    return {
        'video': video_dir / "video.mp4",
        'audio': video_dir / "audio.wav",
        'keypoints': video_dir / "sign.json",
        'intervals': video_dir / "signInterval.json",
        'dir': video_dir
    }


def get_whisper_and_video_paths(video_dir=None):
    """
    Gets paths to Whisper JSON and video from last processing.
    
    Args:
        video_dir: Path to video directory (optional, uses most recent if not specified)
    
    Returns:
        tuple: (whisper_json_path, video_path) or (None, None) if not found
    """
    if video_dir is None:
        video_dir = get_latest_video_dir()
    
    if not video_dir or not video_dir.exists():
        return None, None
    
    whisper_path = video_dir / "audio.json"
    video_path = video_dir / "video.mp4"
    
    return whisper_path, video_path


# ======================================================
# STATE MANAGEMENT
# ======================================================
def save_processing_state(video_dir, metadata=None):
    """
    Saves the state of the last processing.
    
    Args:
        video_dir: Path to processed directory
        metadata: Dictionary with additional information
    """
    state = {
        'timestamp': datetime.now().isoformat(),
        'video_dir': str(video_dir),
        'video_dir_name': video_dir.name,
        'files': {
            'video': str(video_dir / "video.mp4"),
            'keypoints': str(video_dir / "sign.json"),
            'intervals': str(video_dir / "signInterval.json")
        }
    }
    
    if metadata:
        state['metadata'] = metadata
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"✓ State saved in: {STATE_FILE}")
    except Exception as e:
        print(f"⚠ Error saving state: {e}")


def load_processing_state():
    """
    Loads the state of the last processing.
    
    Returns:
        dict: Saved state or None if doesn't exist
    """
    if not STATE_FILE.exists():
        return None
    
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Error loading state: {e}")
        return None


# ======================================================
# VALIDATION
# ======================================================
def validate_environment(check_model=False, check_video=False):
    """
    Validates that the environment has necessary files.
    
    Args:
        check_model: Verify that model files exist
        check_video: Verify that at least one processed video exists
    
    Returns:
        bool: True if everything is correct
    """
    issues = []
    
    # Verify model
    if check_model:
        model_path, words_path = get_model_paths()
        if not model_path.exists():
            issues.append(f"❌ Model not found: {model_path}")
        if not words_path.exists():
            issues.append(f"❌ Words JSON not found: {words_path}")
    
    # Verify video
    if check_video:
        latest = get_latest_video_dir()
        if not latest:
            issues.append(f"❌ No processed videos found in: {DATA_MINER_OUTPUT}")
        else:
            files = get_video_files(latest)
            if not files['keypoints'].exists():
                issues.append(f"❌ Keypoints not found: {files['keypoints']}")
    
    # Show results
    if issues:
        print("\n⚠ DETECTED PROBLEMS:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    return True


# ======================================================
# INITIALIZATION
# ======================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("📋 PROJECT CONFIGURATION")
    print("="*70)
    print(f"\nProject directory: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  • Data miner output: {DATA_MINER_OUTPUT}")
    print(f"  • Models: {MODELS_DIR}")
    print(f"  • Keypoints: {KEYPOINTS_PATH}")
    print(f"  • Predictions: {PREDICTIONS_OUTPUT}")
    print(f"  • Train data: {TRAIN_DATA_DIR}")
    print(f"    - Validated videos: {VALIDATED_VIDEOS_DIR}")
    print(f"    - Validated keypoints: {VALIDATED_KEYPOINTS_DIR}")
    print(f"    - Unknown videos: {UNKNOWN_VIDEOS_DIR}")
    print(f"    - Unknown keypoints: {UNKNOWN_KEYPOINTS_DIR}")
    print(f"\nState file: {STATE_FILE}")
    
    # Create directories
    ensure_directories()
    
    # Verify state
    print("\n" + "="*70)
    print("🔍 ENVIRONMENT VERIFICATION")
    print("="*70)
    
    # Verify models
    model_path, words_path = get_model_paths()
    latest_model_dir = get_latest_model_dir()
    
    if latest_model_dir:
        print(f"\nLatest model directory: {latest_model_dir.name}")
        print(f"  • Model: {model_path.name}")
        print(f"    {'✓ Exists' if model_path.exists() else '✗ Not found'}")
        print(f"  • Words JSON: {words_path.name}")
        print(f"    {'✓ Exists' if words_path.exists() else '✗ Not found'}")
    else:
        print(f"\n✗ No model directories found in {MODELS_DIR}")
        print(f"  (Will be created on first training)")
    
    # Verify processed videos
    latest = get_latest_video_dir()
    if latest:
        print(f"\nLast processed video: {latest.name}")
        files = get_video_files(latest)
        for name, path in files.items():
            if name != 'dir':
                status = "✓" if path.exists() else "✗"
                print(f"  {status} {name}: {path.name}")
    else:
        print("\n✗ No processed videos found")
    
    # Verify saved state
    state = load_processing_state()
    if state:
        print(f"\nSaved state found:")
        print(f"  • Timestamp: {state['timestamp']}")
        print(f"  • Video: {state['video_dir_name']}")
    else:
        print("\n⚠ No saved state")
    
    print("\n" + "="*70)