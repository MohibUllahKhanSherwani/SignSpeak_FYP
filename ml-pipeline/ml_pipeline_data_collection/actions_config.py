import os

config_dir = os.path.dirname(os.path.abspath(__file__))

ACTIONS_FILE = os.path.join(config_dir, "actions.txt")

DATASET_FOLDERS = {
    "laptop": "MP_Data",         
    "mobile": "MP_Data_mobile",  
}
_dataset_setting = os.getenv("SIGNSPEAK_DATASET", "mobile").strip() # Change this line to swap to change dataset
ACTIVE_DATASET = _dataset_setting.lower()
DATA_FOLDER = DATASET_FOLDERS.get(ACTIVE_DATASET, _dataset_setting) or DATASET_FOLDERS["mobile"]

# Where data will be stored
DATA_PATH = DATA_FOLDER if os.path.isabs(DATA_FOLDER) else os.path.join(config_dir, DATA_FOLDER)

# Recording params
SEQUENCE_LENGTH = 60          # number of frames per sequence (2 sec on webcam, 3 secs on mobile)
NUM_SEQUENCES = 20            # how many sequences per action 
FRAME_WAIT_MS = 1           # delay between frames
# Model params
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PREDICTION_THRESHOLD = 0.8      

# Augmentation params
USE_AUGMENTATION = True          
AUGMENTATION_MULTIPLIER = 3      
AUGMENTATION_PROBABILITIES = {
    'horizontal_flip': 0.3,      # Hand-swapping (mirroring) - ESSENTIAL for hand-dominance
    'time_warp': 0.3,            # Speed variations (reduced)
    'spatial_scale': 0.2,        # Distance variations (reduced)
    'spatial_translate': 0.1,    # Position variations (reduced)
    'spatial_rotate': 0.0,       # DISABLED: Too risky for sign recognition
    'add_noise': 0.1,            # Sensor noise (reduced)
    'temporal_crop': 0.2,        # Start/end variations (reduced)
}

def load_actions():
    if not os.path.exists(ACTIONS_FILE):
        raise FileNotFoundError(f"{ACTIONS_FILE} not found. Create it and put one action per line.")

    with open(ACTIONS_FILE, "r", encoding="utf-8") as f:
        actions = [line.strip() for line in f if line.strip()]

    if not actions:
        raise ValueError("actions.txt is empty. Add one action per line.")

    return actions
