const imageInput = document.getElementById('formFile');
const img = document.getElementById('imagePreview');
const placeholderText = document.getElementById('placeholderText');
const imageContainer = document.getElementById('imageContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultContainer = document.getElementById('resultContainer');
const loadingSpinner = document.getElementById('loadingSpinner');

// Image selection and preview
imageInput.addEventListener('change', function(event) {
    const choosedFile = this.files[0];
    if (choosedFile) {
        const reader = new FileReader();
        reader.addEventListener('load', function() {
            img.setAttribute('src', reader.result);
            img.style.display = 'block';
            placeholderText.style.display = 'none';
            imageContainer.classList.add('has-image');
            imageContainer.classList.remove('result-real', 'result-fake');
            analyzeBtn.disabled = false;
            resultContainer.classList.remove('show');
        });
        reader.readAsDataURL(choosedFile);
    }
});

// Image analysis
analyzeBtn.addEventListener('click', async function() {
    const file = imageInput.files[0];
    if (!file) {
        alert('Please select an image first.');
        return;
    }

    // Loading
    loadingSpinner.classList.add('show');
    resultContainer.classList.remove('show');
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/model', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Hide the loading spinner
        loadingSpinner.classList.remove('show');
        
        // gets the probability that image is fake
        const fakeProbability = parseFloat(
            data.fake_probability_percent || data.fake_prob || data.likelihood || data
        );
        const realProbability = 100 - fakeProbability;
        
        // Determine if image is real or fake (threshold at 50%)
        const isFake = fakeProbability > 50;
        
        // If the image is a deepfake, update display
        if (isFake) {
            resultContainer.className = 'result-container show fake';
            imageContainer.classList.add('result-fake');
            imageContainer.classList.remove('result-real');
            document.getElementById('resultLabel').textContent = '⚠️ DEEPFAKE DETECTED';
            document.getElementById('resultPercentage').textContent = `${fakeProbability.toFixed(1)}%`;
            document.getElementById('resultDescription').textContent = 
                `This image is likely AI-generated or manipulated.`;
        } else {
            resultContainer.className = 'result-container show real';
            imageContainer.classList.add('result-real');
            imageContainer.classList.remove('result-fake');
            document.getElementById('resultLabel').textContent = '✅ AUTHENTIC IMAGE';
            document.getElementById('resultPercentage').textContent = `${realProbability.toFixed(1)}%`;
            document.getElementById('resultDescription').textContent = 
                `This image appears to be authentic.`;
        }

    } catch (error) {
        console.error('Error:', error);
        loadingSpinner.classList.remove('show');
        alert('An error occurred while analyzing the image. Please make sure the server is running.');
    } finally {
        analyzeBtn.disabled = false;
    }
});