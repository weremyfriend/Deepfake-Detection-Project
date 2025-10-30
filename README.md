# Deepfake Detection Project

# Requirements
PyTorch:        `pip install torch`
Torchvision:    `pip install torchvision`
Uvicorn:        `pip install uvicorn`
FastAPI:        `pip install fastAPI`
<!--added 10/29/2025 - Walker Hall -->
Flask           `pip install flask`
Python-multipart `pip install python-multipart`

<!-- not needed 10/30/2025-->
# Kaggle Steps: 
~~Users must get their authentication key from Kaggle.com. As the Kaggle API will not run without your key. 
Next step is users must use the Van Gogh Paintings dataset from following Kaggle user: ipythonx/van-gogh-paintings (ipythonx is the users name, and Van-gogh-paintings is name of dataset used in this project)~~


# Run FastAPI server
To run the server you must run the bash command
`uvicorn app.defaker-test:app --host 127.0.0.1 --port 5000 --reload`

This will run the server using FastAPI and ensure the Python file can connect to the JavaScript file via an endpoint.

<!--added 10/29/2025 - Walker Hall -->
# UI
Navigate to `http://127.0.0.1:5000/ui/deepfake_defake_ui.html`

<!-- Added 10/30/2025 -->
# Training
Find a good source for training data for deepfake images. A Google search shows promising results with sources that include over 1000 thousand images.
Examples:
<br>
https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset
<br>
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Add the training and testing images to their respective folders. 

Adding images will make the model retrain on restart
Delete the contents in the saved_models folder to start from scratch if a model/optimizer is present