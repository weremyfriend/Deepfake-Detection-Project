let imageInput = document.getElementById("formFile");
const img = document.querySelector('#imagePreview');
// img.style.display = "none";
imageInput.addEventListener("change", function(event) {
    event.preventDefault();
    img.style.display = "block";
    const choosedFile = this.files[0];
    if (choosedFile) {
        const reader = new FileReader(); //FileReader is a predefined function of JS
        reader.addEventListener('load', function() {
            img.setAttribute('src', reader
                .result); // [1] because we have 2 images with id avtar,
        });
        reader.readAsDataURL(choosedFile);

        // Send the image to the Python backend
        const formData = new FormData();
        formData.append('image', choosedFile);
        event.preventDefault();
        fetch('http://localhost:5000/upload', { // Update the URL to your Python endpoint
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json(); // Assuming Python responds with JSON
        })
        .then(data => {
            console.log('Success:', data);
            if (data.probability_fake !== undefined) {
                resultDisplay.innerText = `Probability the image is fake: ${data.probability_fake.toFixed(2)}%`;
            } else {
                resultDisplay.innerText = "Unable to determine the probability.";
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDisplay.innerText = "An error occurred while processing the image.";
        });
    }
    event.preventDefault();
})