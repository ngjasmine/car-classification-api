# car-classification-api

## Introduction
Welcome to the car classification API! Using this repository, you will be able to upload images, and get back the classification of car model as determined by the model.


## Quick Start Guide
1. Fork and clone this repository.

2. Docker is required to be installed on your local machine. Ensure the docker daemon is running, and that you are logged into docker:
    ```
    docker login
    ```

3. Build docker image using the following command:
    ``` 
    docker build -t car-classification-api:v1 .
    ```

4. Run docker container
    ```
    docker run -d -p 8000:8000 car-classification-api:v1
    ```
* `-d` flag indicates running the container in detached mode (so that we can use the terminal for other actions; see below).
* `-p` flag  forwards traffic from local machine's 8000 port to container's port 8000 - this allows us to access the FastAPI app from outside the container.

5. Check that the FastAPI app is running by visiting http://127.0.0.1:8000/docs on a web browser on your local machine. For a single image you may choose to upload it using the interface here. 

6. Run the batch upload script for uploading multiple images with the folder path as an argument:
    ```
    python app/upload_images.py "path_to_images_folder"
    ```
    You should get back a similar output in the terminal:
    ```
    {
        "predictions": {
            "001884.jpg": {
                "class_id": 166,
                "class_name": "Mitsubishi Lancer Sedan 2012"
            },
            "001847.jpg": {
                "class_id": 63,
                "class_name": "Chevrolet Malibu Hybrid Sedan 2010"
            },
            "001853.jpg": {
                "class_id": 166,
                "class_name": "Mitsubishi Lancer Sedan 2012"
            }
        }
    }
    ```
    The output is also saved as a json file in output_dir as specified in upload_images.py.

## Background
The following section captures the workflow used in developing this project:

1. **Literature Review** - a brief literature review was carried out in order to identify suitable classification models for this task. The main point considered when picking 3 models for carrying out transfer learning (ie. retaining the weights feature extraction layers and only training the classification layer) was local computing resources available - this was limited, and knowing the dataset available - 8144 images in train set and 8041 images in test set - there was a focus on finding models with a smaller size and less parameters. There was no requirement given for inference speed, so this was not taken into consideration. 3 models identified were efficientnet_b0, mobilenet_v2 and resnet50.
    * With efficientnet_b0, a separate set of experiment was carried out where additional data preprocessing was carried out on train data with a package called autoaugment
    * With resnet50, both Adam and SGD optimizers were used 

2. **Exploratory Data Analysis** - the notebook EDA.ipynb was used to carry out exploratory data analysis on the dataset.

2. **Test results for transfer learning:**

    | Model                      | Train Loss (20th Epoch) | Train Acc (20th Epoch) | Val Loss (20th Epoch) | Val Acc (20th Epoch) | Test Loss | Test Acc |
    |----------------------------|------------------------|------------------------|-----------------------|----------------------|-----------|----------|
    | efficientnet_b0_autoaugment | 2.555                  | 45.56                  | 3.025                 | 31.61                | 3.0358    | 32.36    |
    | efficientnet_b0           | 1.774                  | 64.13                  | 2.903                 | 33.39                | 2.9106    | 34.04    |
    | mobilenet_v2             | 1.9346                 | 62.7                   | 3.0388                | 32.66                | 3.0828    | 31.75    |
    | resnet50_adam            | 1.2276                 | 84.62                  | 2.6477                | 38.12                | 2.6355    | 39.87    |
    | resnet50_sgd             | 4.4033                 | 26.14                  | 4.4806                | 16.21                | 4.4848    | 15.52    |

    Since the 1st 4 models utilized Adam optimizer, it would be fair to compare those and leave out the resnet50_sgd model which used SGD optimizer (in any case its performance for this transfer learning task was not great). The best performning model based on val loss / val accuracy was resnet50.
    * Local logging was carried out during transfer learning

3. **Hyperparameter tuning for resnet50**
    * This was carried out using Ray Tune
    * It encompasses experiment/trial logging therefore local logging was not invoked here
    * The following search space was used:
    ```
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2), 
        "batch_size": tune.choice([16, 32, 64]),
        "num_bottlenecks_to_unfreeze": tune.choice([1, 2]), 
        "num_classes": len(datasets.ImageFolder(train_dir).classes),
        "epochs": 15 
    }
    ```
    * And the total number of trials was set to 6, again due to local compute constraints.
    * For graphs of loss against epochs and accuracy against epochs, see the notebook hyperparameter-tuning-resnet50.ipynb.

4. **Loading best model, evaluating on test data, and saving best model** - this was done in the notebook load-eval-save-best-model.ipynb. Best model was defined as having the lowest val loss; the identified best model had the following config:
    ```
    {'lr': 0.00023835124360836502, 'batch_size': 32, 'num_bottlenecks_to_unfreeze': 2, 'num_classes': 196, 'epochs': 15}
    ```
    The test accuracy obtained was 68.70% - a great improvement over 39.87% obtained from transfer learning, showing that there is space for further improvement with additional computational resource and time.

5. **FastAPI app and Docker** - A FastAPI app was developed in order to allow users to get predictions from an image/folder of images. This was containerized using Docker. The instructions to use the app is as above in [Quick Start Guide](#quick-start-guide).



## Assumptions
- This README.md is specifically written for users who want to use the FastAPI app for inference; and not for developers who want to further train the model - though that would be possible by making a copy of the hyperparameter-tuning notebook and the load-eval-save-best-model notebook.
- Metric to optimize for is best accuracy: this was the accuracy 
- No requirement for inference speed
- FastAPI is only required to be hosted locally
- User is uploading an image/images from a local directory rather than from the web (eg. S3 bucket)




## Additonal Challenge - Configure a Pipeline
Github's CI/CD could be a suitable tool to use 