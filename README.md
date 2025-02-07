# car-classification-api

## Introduction
Welcome to the car classification API! Using this repository, you will be able to upload an image / a folder of images, and get back the classification of car model as determined by the model.


## Quick Start Guide for Users
NOTE: the deployment environment must have at least 3 GB of available storage: 2.5 GB for the Docker image, and additional space for container layers, logs, and temporary files during execution.
1. Clone this repository:
    ```
    git clone https://github.com/ngjasmine/car-classification-api.git
    cd car-classification-api
    ```

2. Docker is required to be installed on your local machine. If it is not, [install Docker](https://docs.docker.com/engine/install/). Ensure the docker daemon is running.

3. Build docker image using the following command:
    ``` 
    docker build -t car-classification-api:v1 .
    ```

4. Obtain the local path to the folder of images you want to perform inference on. This will be used in the next step, as `<path_to_folder_for_inference>`.

5. Run docker container using the following command:
    ```
    docker run -d -p 8000:8000 \
    -v "$(pwd)/app:/car-classification-api/app" \
    -v "<path_to_folder_for_inference>:/app/data/for-prediction" \
    -e TZ=Asia/Singapore \
    car-classification-api:v1
    ```
* `-d` flag indicates running the container in detached mode (so that we can use the terminal for other actions; see step 7 below).
* `-p` flag  forwards traffic from local machine's 8000 port to container's port 8000

6. Check that the FastAPI app is running by visiting http://127.0.0.1:8000/docs on a web browser on your local machine. For a single image you may choose to upload it using the interface here under POST.

7. Obtain the container ID by running the command below. This will be used in the next step, as `<container_id>`.
```
docker ps
```

8. In order to carry out inference on the folder of images, :
    ```
    docker exec -it <container_id> python app/upload_images.py "/app/data/for-prediction"
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
    The output is also saved as a json file in output_dir as specified in upload_images.py; this is currently set as `car-classification-api/app/predictions`.

9. Stop the docker container and other useful commands:
    ```
    docker ps  # List running containers
    docker stop <container_id>  # Stop a running container
    docker rm <container_id>  # Remove a container
    docker rmi car-classification-api:v1  # Remove the image
    ```

## Background
The following section captures the workflow used in developing this project:

1. **Literature Review** - a brief literature review was carried out in order to identify suitable pre-trained open-source classification models for this computer vision task. The main point considered when picking 3 models for carrying out transfer learning (ie. retaining the weights feature extraction layers and only training the classification layer) was local computing resources available - this was limited, and knowing the dataset to be used - 8144 images in train set and 8041 images in test set - there was a focus on finding models with a smaller size and fewer parameters. There was no requirement given for inference speed, so this was not taken into consideration. 3 models identified were `efficientnet_b0`, `mobilenet_v2` and `resnet50`. These models are standard, pretrained models available from torchvision.
* With `efficientnet_b0`, a separate set of experiment was carried out where additional data preprocessing was carried out on train data with a package called autoaugment
* With `resnet50`, both Adam and SGD optimizers were used 

2. **Exploratory Data Analysis** - the notebook EDA.ipynb was used to carry out exploratory data analysis on the dataset.

3. **Test results for transfer learning:**

    | Model                      | Train Loss (20th Epoch) | Train Acc (20th Epoch) | Val Loss (20th Epoch) | Val Acc (20th Epoch) | Test Loss | Test Acc |
    |----------------------------|------------------------|------------------------|-----------------------|----------------------|-----------|----------|
    | efficientnet_b0_autoaugment | 2.555                  | 45.56                  | 3.025                 | 31.61                | 3.0358    | 32.36    |
    | efficientnet_b0           | 1.774                  | 64.13                  | 2.903                 | 33.39                | 2.9106    | 34.04    |
    | mobilenet_v2             | 1.9346                 | 62.7                   | 3.0388                | 32.66                | 3.0828    | 31.75    |
    | resnet50_adam            | 1.2276                 | 84.62                  | 2.6477                | 38.12                | 2.6355    | 39.87    |
    | resnet50_sgd             | 4.4033                 | 26.14                  | 4.4806                | 16.21                | 4.4848    | 15.52    |

    Since the 1st 4 models utilized Adam optimizer, it would be fair to compare those and leave out the resnet50_sgd model which used SGD optimizer (in any case its performance for this transfer learning task was not great). The best performning model based on val loss / val accuracy was resnet50, achieving a test accuracy score of **39.87%**. Therefore it was picked as the model to carry out hyperparameter tuning on.
* The notebooks used for transfer learning with the different models start with *'transfer-learning-...'*.ipynb
* For explanations regarding *data pre-processing steps* and *model training process*, refer to `transfer-learning-resnet50-adam.ipynb`.
* Local logging was carried out during transfer learning

4. **Hyperparameter tuning for resnet50 model**
* This was carried out using Ray Tune
* Ray tune has experiment/trial logging built in therefore local logging was not invoked here
* The following search space was used, with a modest number of values per hyperparamter due to local compute constraints:
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
* For final training and val results (graphs of loss against epochs and accuracy against epochs), see the notebook `hyperparameter-tuning-resnet50.ipynb`.

5. **Loading best model, evaluating on test data, and saving best model** - this was done in the notebook `load-eval-save-best-model.ipynb`. Best model was defined as having the lowest val loss; the identified best model had the following config:
    ```
    {'lr': 0.00023835124360836502, 'batch_size': 32, 'num_bottlenecks_to_unfreeze': 2, 'num_classes': 196, 'epochs': 15}
    ```
* The test accuracy obtained was **68.70%** - a great improvement over 39.87% obtained from transfer learning, showing that there is space for further fine-tuning of the model with additional computational resource and time.
* Final confusion matrix and confusion report are available in this notebook.
* Best model is saved to `models/` folder with file name `best_resnet50_model.pth`

6. **FastAPI app and Docker** - A FastAPI app was developed in order to allow users to get predictions from an image/folder of images. This was containerized using Docker. The instructions to use the app is as above in [Quick Start Guide](#quick-start-guide).
* Source code files for FastAPI app can be found in app/
* Dockerfile is available in project root


## Assumptions
- This README.md is specifically written for users who want to use the FastAPI app for inference; and not for developers who want to further train the model (though that would be possible by forking and cloning this repository).
- Metric to optimize for is best accuracy.
- No requirement for inference speed.
- FastAPI app is only required to be hosted locally.
- User is uploading an image/images from a local directory rather than from the web (eg. S3 bucket).
- User is aware that dataset is of cars available in the United States and not very applicable in local context; it would be useful to have a local dataset if this solution is to be used locally.
- There is low traffic to the app (eg. 20 - 100 users), so Nginx was not used which would help with load balancing on multiple uvicorn instances
- The target deployment machine has at least 3 GB of available storage: 2.5 GB for the Docker image, and additional space for container layers, logs, and temporary files during execution.


## Additonal Challenge - Configure a Pipeline
To streamline training and evaluation, versioning and deployment, a CI/CD pipeline can be implemented using GitHub Actions, which consists of the following key stages:

1. **Triggering Events**  
The pipeline can be triggered when:
    * A push is made to the main branch.
    * A pull request is opened against the main branch.
    * A new tag is created for versioning.

2. **Preprocessing & Training**
* A GitHub Actions workflow runs the training script when changes are pushed to a training-related branch (e.g., model-training).
* Training script uses Ray Tune to optimize hyperparameters.
* Training script logs experiments to WandB or MLFlow.
* Training script saves the trained model with versioning - model can be saved locally or using a registry, see point 4 below.

3. **Evaluation & Testing**
* Loads the best model based on selected metric eg. validation loss.
* Computes test accuracy and confusion matrix.
* If test accuracy surpasses a defined threshold (e.g., 65%), the model is approved for deployment.

4. **Model Versioning & Storage**
* Stores the model in a model registry (e.g., MLFlow, WandB Artifacts, S3, or a custom storage).
* Assigns a version tag to the model.

5. **Containerization & Deployment**
* Builds a Docker image with the FastAPI app and trained model.
* Pushes the Docker image to a container registry (Docker Hub, Harbor, or AWS ECR).
* Deploys the API container on a local machine or cloud instance.