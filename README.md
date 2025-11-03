# Deepfake Detection Project

## 1. Requirements
Python 3.14.0

pip 25.3


PyTorch:        `pip install torch`

Torchvision:    `pip install torchvision`

Uvicorn:        `pip install uvicorn`

FastAPI:        `pip install fastAPI`

PIL:        `pip install pillow`

Python-multipart `pip install python-multipart`

Microsoft Visual C++ Redistributable (2015-2022) `https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version`

### (optional)
**For NVIDIA GPUs**

CUDA PyTorch        `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

<!-- 11/2/2025
**For testing**
Pytest  `pip install pytest`
Pytest-cov `pip install pytest-cov`
FastAPI (all dependencies) `pip install fastapi[all]`
-->



<!-- Added 10/30/2025 -->
## 2. Training
There are good sources for training data with deepfake faces on Kaggle.

Examples:

a. https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset

b. https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

>Add the training and testing images to their respective folders. <br>
>Adding images will make the model retrain on restart <br>
>Delete the contents in the saved_models folder to start from scratch if a model/optimizer is present



## 3. Run the FastAPI server
To run the server you must be in the root of the proram <br>
`uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload`

This will run the server using FastAPI and ensure the Python file can connect to the JavaScript file via an endpoint.

<!--added 10/29/2025 - Walker Hall -->
## 4. UI
Navigate to `http://127.0.0.1:5000/ui/index.html` on the machine that's running the server

<!-- 11/2/2025
## Testing (11/2/2025)
Run all unit tests `python -m unittest discover tests` or `pytest tests/ -v`
-->

## Todo
1. Fix UI routing
2. Improve accuracy and performance of model from scratch
3. Include different models
4. Linux support with AMD GPUs  (ROCm)
5. Containerization/deployable on a server
6. Unrestricted file upload, CORS middleware, fix internal HTTP errors to prevent information disclosure, authentication, delete uploaded files, encrypt user data, and HTTPS.
7. Detect deepfakes from videos