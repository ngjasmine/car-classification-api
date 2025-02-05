import os
import argparse
import requests
import json

parser = argparse.ArgumentParser(description="Upload all images from specified\
                                 folder to the FastAI server.")
parser.add_argument("folder_path", type=str, help="Path to the folder \
                    containing images for inference.")

args = parser.parse_args()
folder_path = args.folder_path

# FastAPI URL
url = "http://127.0.0.1:8000/predict/"

# Check if the folder exists
if not os.path.isdir(folder_path):
    print(f"Error: The specified folder '{folder_path}' does not exist.")
    exit(1)

# Obtain image files in the folder
image_files = [
    ("files", (file, open(os.path.join(folder_path, file), "rb"), "image/jpeg"))
    for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))
]

if not image_files:
    print("No image files found in the specified folder.")
    exit(1)

# Send the request
response = requests.post(url, files=image_files)

# Pretty print the response
response_json = response.json()
print(json.dumps(response_json, indent=4))