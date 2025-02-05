import os
import pytest
import yaml
import torch
import tempfile
import shutil
import pickle
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.utils import (
    check_file_extensions, get_image_resolutions, channel_count,
    load_config, create_logger, load_best_resnet50_model
)

def test_check_file_extensions():
    """
    Tests that check_file_extensions correctly identifies unique file extensions
    in a given directory, including subdirectories.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        open(os.path.join(temp_dir, "test1.jpg"), "w").close()
        open(os.path.join(temp_dir, "test2.png"), "w").close()
        os.mkdir(os.path.join(temp_dir, "subdir"))
        open(os.path.join(temp_dir, "subdir", "test3.jpeg"), "w").close()
        extensions = check_file_extensions(temp_dir)
        assert extensions == {".jpg", ".png", ".jpeg"}

def test_get_image_resolutions():
    """
    Tests that get_image_resolutions correctly extracts image dimensions from image files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = os.path.join(temp_dir, "test.jpg")
        img = Image.new("RGB", (100, 200))
        img.save(img_path)
        resolutions = get_image_resolutions(temp_dir)
        assert resolutions == [(100, 200)]

def test_channel_count():
    """
    Tests that channel_count correctly identifies image formats and their respective
    channel distributions.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = os.path.join(temp_dir, "test.png")
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        image_formats, channel_counts = channel_count(temp_dir)
        assert image_formats == {"PNG": 1}
        assert channel_counts == {3: 1}

def test_load_config():
    """
    Tests that load_config correctly loads a YAML configuration file.
    """
    config_data = {"test_key": "test_value"}
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        yaml.dump(config_data, temp_file)
        temp_file_path = temp_file.name
    try:
        config = load_config(temp_file_path)
        assert config == config_data
    finally:
        os.remove(temp_file_path)

def test_create_logger():
    """
    Tests that create_logger initializes a logger with the correct configuration
    from a YAML configuration file.
    """
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "": {
                "level": "INFO",
                "handlers": ["h1", "h2"],
                "propagate": True
            }
        },
        "handlers": {
            "h1": {
                "class": "rich.logging.RichHandler",
                "level": "INFO",
                "formatter": "brief"
            },
            "h2": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "precise",
                "filename": "training.log",
                "mode": "a",
                "encoding": "utf8"
            }
        },
        "formatters": {
            "brief": {
                "format": "%(levelname)s - %(message)s"
            },
            "precise": {
                "format": "%(asctime)s : %(levelname)s : %(filename)s, line %(lineno)s : %(message)s"
            }
        }
    }
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
        yaml.dump(logger_config, temp_file)
        temp_file_path = temp_file.name
    try:
        log_file = "test.log"
        logger = create_logger(temp_file_path, log_file)
        assert logger is not None
    finally:
        os.remove(temp_file_path)
        if os.path.exists(log_file):
            os.remove(log_file)

@patch("torchvision.models.resnet50")
def test_load_best_resnet50_model(mock_resnet):
    """
    Tests that load_best_resnet50_model correctly selects the best model checkpoint
    based on validation loss and loads the corresponding model.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        exp_dir = os.path.join(temp_dir, "experiment1")
        os.makedirs(exp_dir)
        trial_dir = os.path.join(exp_dir, "trial1")
        os.makedirs(trial_dir)
        checkpoint_dir = os.path.join(trial_dir, "checkpoint_epoch_1")
        os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_epoch_1.pkl")
        checkpoint_data = {"val_loss": 0.1, "config": {"num_classes": 10}, "model_state_dict": {}}
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
        model, config = load_best_resnet50_model(temp_dir)
        assert isinstance(model, MagicMock)
        assert config == {"num_classes": 10}
