from pathlib import Path

# Get root directory
def get_root() -> Path:
    return Path(__file__).resolve().parent.parent

# Get data directory
def get_data():
    return get_root() / 'data'

# Get model directory
def get_model():
    return get_root() / 'model'

# Get config directory
def get_config():
    return get_root() / 'config'