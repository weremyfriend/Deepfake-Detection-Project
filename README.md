# Deepfake Detection Project

The objective of this project is to design and implement a deepfake detection system using a Convolutional Neural Network (CNN). The system aims to detect whether an image is a deepfake image of a face (synthetically made or modified) or is a genuine, unaltered image of a face. It does this by analyzing facial and pixel-level inconsistencies and outputs a probability score indicating the likelihood that an image is fake. 

## 1. Requirements
**Python 3.14.0**

**pip 25.3**


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



## 2. Training
Training data should be a diverse source of deepfake and real faces. Kaggle is a great source.

Examples:

a. https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset

b. https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

**Add the images to train and test the model to dataset/train and dataset/test respectively.**



Note: 
Adding images will make the model retrain on restart! <br>




## 3. Run the FastAPI server
To run the server you must be in the root of the application and execute this command: <br>
`uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload`

Note: This will run the server using FastAPI and ensure the Python file can connect to the JavaScript file via an endpoint.

## 4. UI
To see the UI, navigate to `http://127.0.0.1:5000/ui/index.html` on the machine that's running the server

<!-- 11/2/2025
## Testing (11/2/2025)
Run all unit tests `python -m unittest discover tests` or `pytest tests/ -v`
-->

## Todo
1. Containerization/deployable on a server
2. Unrestricted file upload, CORS middleware, fix internal HTTP errors to prevent information disclosure, authentication, delete uploaded files, encrypt user data, and HTTPS.
3. Detect deepfakes from videos