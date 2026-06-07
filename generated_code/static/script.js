document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultCard = document.getElementById('resultCard');
    const statusIndicator = document.getElementById('statusIndicator');
    const resultLabel = document.getElementById('resultLabel');
    const resultConfidence = document.getElementById('resultConfidence');
    const resultText = document.getElementById('resultText');
    const resultDetails = document.getElementById('resultDetails');
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

    // Analyze Another Button
    analyzeAnotherBtn.addEventListener('click', () => {
        uploadForm.reset();
        imagePreview.style.display = 'none';
        imagePreview.innerHTML = '';
        dropZone.style.display = 'flex';
        resultCard.classList.add('hidden');
    });

    // Drag and Drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drop-zone--over');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('drop-zone--over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drop-zone--over');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updateThumbnail(dropZone, fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            updateThumbnail(dropZone, fileInput.files[0]);
        }
    });

    function updateThumbnail(dropZone, file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                imagePreview.style.display = 'flex';
                imagePreview.innerHTML = `<img src="${reader.result}" alt="Preview">`;
                dropZone.style.display = 'none';
            };
        }
    }

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(uploadForm);
        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;
        resultCard.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showResult(data);
            } else {
                alert(`Error: ${data.error || 'Something went wrong'}`);
            }
        } catch (error) {
            alert(`Error connecting to server: ${error.message}`);
            console.error(error);
        } finally {
            analyzeBtn.textContent = 'Analyze Meme';
            analyzeBtn.disabled = false;
        }
    });

    function showResult(data) {
        resultCard.classList.remove('hidden');
        resultLabel.textContent = data.label;
        resultConfidence.textContent = `Confidence: ${data.confidence}`;

        // Display Extracted Text
        if (data.extracted_text) {
            resultText.textContent = `Extracted Text: "${data.extracted_text}"`;
            resultText.style.display = 'block';
        } else {
            resultText.style.display = 'none';
        }

        // Display Details
        resultDetails.style.display = 'none';

        // Display Logic Gate breakdown
        const logicBreakdown = document.getElementById('logicBreakdown');
        if (data.image_label && data.text_label && logicBreakdown) {
            logicBreakdown.style.display = 'block';

            const imageStatus = document.getElementById('imageStatus');
            const textStatus = document.getElementById('textStatus');

            imageStatus.textContent = `Image: ${data.image_label}`;
            textStatus.textContent = `Text: ${data.text_label}`;

            imageStatus.style.color = (data.image_label === 'Hateful') ? 'var(--danger)' : 'var(--success)';
            textStatus.style.color = (data.text_label === 'Hateful') ? 'var(--danger)' : 'var(--success)';
        } else if (logicBreakdown) {
            logicBreakdown.style.display = 'none';
        }

        statusIndicator.classList.remove('status-safe', 'status-hateful');
        if (data.label === 'Hateful') {
            statusIndicator.classList.add('status-hateful');
            resultLabel.style.color = 'var(--danger)';
        } else {
            statusIndicator.classList.add('status-safe');
            resultLabel.style.color = 'var(--success)';
        }

        // Speak Result
        speakResult(data);
    }

    function speakResult(data) {
        if ('speechSynthesis' in window) {
            const textToSpeak = `The meme is classified as ${data.label}. Confidence is ${data.confidence}. ${data.details ? data.details : ''}`;
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.rate = 1;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
        }
    }
});
