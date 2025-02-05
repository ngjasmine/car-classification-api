import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from typing import Dict, Any
import logging
import logging.config
from torchvision import models
from pathlib import Path
import pickle
import torch

def check_file_extensions(data_dir):
    """
    Check and display all unique file extensions in the dataset.

    Args:
        dataset_dir (str): Path to the dataset containing class subdirectories.

    Returns:
        set: A set of unique file extensions in the dataset.
    """
    extensions = set()
    for dir, _, files in os.walk(data_dir):
        for file in files:
            if not file.startswith('.'):  # Exclude hidden files
                ext = os.path.splitext(file)[-1].lower()  # Get file extension and normalize to lowercase
                extensions.add(ext)
    return extensions

def get_image_resolutions(data_dir):
    """
    Retrieve the resolutions (width and height) of all images in a specified directory.

    This function recursively traverses a directory and collects the resolutions 
    (width, height) of all image files with extensions `.png`, `.jpg`, or `.jpeg`.

    Args:
        data_dir (str or Path): Path to the directory containing image files.

    Returns:
        list of tuple: A list of tuples, where each tuple contains the width and height 
        of an image (e.g., [(1920, 1080), (1280, 720), ...]).
    
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PIL.UnidentifiedImageError: If a file cannot be opened as an image.
        """
    resolutions = []
    for root, dir, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    resolutions.append(img.size)  # img.size is a tuple (width, height)
    return resolutions

def channel_count(data_dir):
    """
    Analyzes image files in a given directory to count image formats and channel configurations.

    Args:
        data_dir (str): The directory path containing the image files to analyze.

    Returns:
        tuple: 
            - image_formats (dict): A dictionary with image formats as keys and their counts as values.
            - channel_counts (dict): A dictionary with the number of channels as keys and their counts as values.

    Supported Image Formats:
        - .png
        - .jpg
        - .jpeg

    Channel Modes:
        - 'L': 1 channel (grayscale)
        - 'RGB', 'YCbCr': 3 channels
        - 'RGBA', 'CMYK': 4 channels
        - 'P': 3 channels (palette-based)
        - '1': 1 channel (binary)

    Prints:
        - Unhandled modes with filename for debugging.
    """
        
    image_formats = {}
    channel_counts = {}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    # Format
                    format = img.format
                    image_formats[format] = image_formats.get(format, 0) + 1
                    
                    # Channels
                    mode = img.mode
                    if mode == 'L':
                        channels = 1
                    elif mode in ['RGB', 'YCbCr']:
                        channels = 3
                    elif mode == 'RGBA':
                        channels = 4
                    elif mode == 'CMYK':
                        channels = 4
                    elif mode == 'P':  # Palette-based images are usually treated as 3-channel images
                        channels = 3
                    elif mode == '1':  # Binary images (black and white)
                        channels = 1
                    else:
                        channels = None  # Unknown or unhandled mode

                    if channels is not None:
                        channel_counts[channels] = channel_counts.get(channels, 0) + 1
                    else:
                        print(f"Unhandled mode: {mode} in file {file}")

    return image_formats, channel_counts

def display_random_images(dataset_dir, num_classes=20, grid_size=(4,5), fig_size=(20, 15)):

    """
    Display random images from randomly selected classes in a dataset.

    Args:
        dataset_dir (str): Path to the dataset containing class subdirectories.
        num_classes (int): Number of classes to sample images from.
        grid_size (tuple): Tuple indicating the grid layout (rows, columns).
        fig_size (tuple): Figure size for the plot.

    Returns:
        None
    """
    def display_image_in_subplot(class_path, class_label, subplot_num):
        """Display a random image from a given class directory."""
        files = [f for f in os.listdir(class_path) if not f.startswith('.')]  # Exclude hidden files
        if files:
            img_name = random.choice(files)
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            ax = plt.subplot(grid_size[0], grid_size[1], subplot_num)
            plt.imshow(img)
            plt.title(class_label, fontsize=8)
            plt.axis('off')
        else:
            print(f"No images found in {class_path}")

    # Get all class labels (subdirectories in the dataset directory)
    all_class_labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Randomly pick classes to display
    random_class_labels = random.sample(all_class_labels, min(num_classes, len(all_class_labels)))

    # Prepare the figure
    plt.figure(figsize=fig_size)

    # Display one random image from each selected class
    for subplot_num, class_label in enumerate(random_class_labels, start=1):
        class_path = os.path.join(dataset_dir, class_label)
        display_image_in_subplot(class_path, class_label, subplot_num)

    plt.tight_layout()
    plt.show()

def load_config(path: str) -> Dict[str, Any]:
	"""
	Loads a configuration from a YAML file.
	
	:path (str): The path to the YAML file that contains the configuration
	:return: The configuration data loaded from the YAML file as a variable called config.
	"""
	with open(path, 'r') as file:
		config = yaml.safe_load(file)
	return config

def create_logger(logger_config_path, log_file_path):
	"""
	Initializes a logger with a dynamically updated log file path.
	
	:param logger_config_path: Path to the logger configuration file.
	:type logger_config_path: str
    :param filename: Full path to the log file where logs should be written.
    :type filename: str
    :return: A logger instance configured as specified in the YAML file.
	"""
	logger_config = load_config(logger_config_path)
	
	logger_config['handlers']['h2']['filename'] = log_file_path
	
	logging.config.dictConfig(logger_config)

	return logging.getLogger(__name__)

def load_best_resnet50_model(ray_results_dir):
    """
    Load the best restnet50 model based on validation loss from Ray Tune results.

    Args:
        ray_results_dir (str or Path): Path to the Ray results directory.

    Returns:
        torch.nn.Module: The model loaded with the best checkpoint.
        dict: The configuration of the best trial.
    """
    ray_results_dir = Path(ray_results_dir)
    best_val_loss = float('inf')
    best_checkpoint_file = None
    best_config = None

    # Traverse experiments
    for experiment_dir in ray_results_dir.iterdir():
        if not experiment_dir.is_dir():
            continue  # Skip non-directory files like tuner.pkl

        # Traverse trials within each experiment
        for trial_dir in experiment_dir.iterdir():
            if not trial_dir.is_dir():
                continue  # Skip non-directory files

            # Traverse checkpoints within each trial
            for checkpoint_dir in trial_dir.glob("checkpoint_epoch_*"):
                if not checkpoint_dir.is_dir():
                    continue  # Skip non-directory files

                checkpoint_file = checkpoint_dir / f"{checkpoint_dir.name}.pkl"
                if not checkpoint_file.exists():
                    continue  # Skip if the checkpoint file doesn't exist

                # Load checkpoint data
                with open(checkpoint_file, "rb") as fp:
                    checkpoint_data = pickle.load(fp)

                # Check for val_loss
                if "val_loss" not in checkpoint_data:
                    print(f"Skipping {checkpoint_file}: 'val_loss' not found")
                    continue

                val_loss = checkpoint_data["val_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_file = checkpoint_file
                    best_config = checkpoint_data["config"]

    if best_checkpoint_file is None:
        raise ValueError("No valid checkpoint with 'val_loss' found.")

    # Load the best checkpoint data
    with open(best_checkpoint_file, "rb") as fp:
        best_checkpoint_data = pickle.load(fp)

    # Initialize the resnet50 model
    model = models.resnet50(weights='DEFAULT')
    # Get input features of the fully connected layer
    num_ftrs = model.fc.in_features
    
    # Replace the fully connected layer with correct number of output classes
    model.fc = torch.nn.Linear(num_ftrs, best_config["num_classes"])

    model.load_state_dict(best_checkpoint_data["model_state_dict"])
    model.eval()  # prevents unintended train mode

    return model, best_config