import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def check_file_extensions(dataset_dir):
    """
    Check and display all unique file extensions in the dataset.

    Args:
        dataset_dir (str): Path to the dataset containing class subdirectories.

    Returns:
        set: A set of unique file extensions in the dataset.
    """
    extensions = set()
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.startswith('.'):  # Exclude hidden files
                ext = os.path.splitext(file)[-1].lower()  # Get file extension and normalize to lowercase
                extensions.add(ext)
    return extensions

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