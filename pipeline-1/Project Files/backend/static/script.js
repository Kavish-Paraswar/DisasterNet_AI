document.addEventListener('DOMContentLoaded', () => {

    // --- Tab Switching Logic ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    let currentMode = 'upload'; // 'upload' or 'webcam'
    let currentFile = null;     // Stores the uploaded file
    let webcamStream = null;    // Stores the active webcam stream

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active to clicked tab
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');

            if (targetId === 'upload-tab') {
                currentMode = 'upload';
                stopWebcam();
                updateAnalyzeButtonState();
            } else {
                currentMode = 'webcam';
                updateAnalyzeButtonState();
            }
        });
    });

    // --- Upload Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('upload-preview-container');
    const previewImg = document.getElementById('upload-preview-img');
    const clearUploadBtn = document.getElementById('clear-upload');

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // File Input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            updateAnalyzeButtonState();
        };
        reader.readAsDataURL(file);
    }

    clearUploadBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        updateAnalyzeButtonState();
    });


    // --- Webcam Logic ---
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const webcamPreview = document.getElementById('webcam-preview-img');
    
    const startWebcamBtn = document.getElementById('start-webcam');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');

    let base64Capture = null;

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            video.srcObject = webcamStream;
            video.classList.remove('hidden');
            webcamPreview.classList.add('hidden');
            
            startWebcamBtn.classList.add('hidden');
            captureBtn.classList.remove('hidden');
            retakeBtn.classList.add('hidden');
            base64Capture = null;
            updateAnalyzeButtonState();
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access the webcam. Please ensure permissions are granted.");
        }
    }

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        video.srcObject = null;
        startWebcamBtn.classList.remove('hidden');
        captureBtn.classList.add('hidden');
        retakeBtn.classList.add('hidden');
        video.classList.remove('hidden');
        webcamPreview.classList.add('hidden');
        base64Capture = null;
    }

    startWebcamBtn.addEventListener('click', startWebcam);

    captureBtn.addEventListener('click', () => {
        // Draw video frame to canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64
        base64Capture = canvas.toDataURL('image/jpeg');
        
        // Show preview
        webcamPreview.src = base64Capture;
        video.classList.add('hidden');
        webcamPreview.classList.remove('hidden');
        
        // Toggle buttons
        captureBtn.classList.add('hidden');
        retakeBtn.classList.remove('hidden');
        
        // Stop the active stream to save resources since we have the photo
        webcamStream.getTracks().forEach(track => track.stop());
        
        updateAnalyzeButtonState();
    });

    retakeBtn.addEventListener('click', startWebcam);


    // --- Global Logic ---
    const analyzeBtn = document.getElementById('analyze-btn');

    function updateAnalyzeButtonState() {
        if (currentMode === 'upload' && currentFile) {
            analyzeBtn.disabled = false;
        } else if (currentMode === 'webcam' && base64Capture) {
            analyzeBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
        }
    }

    // --- API Request & UI Updates ---
    const resultContent = document.getElementById('result-content');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultCard = document.getElementById('result-card');
    
    // Elements to update
    const predictionText = document.getElementById('prediction-text');
    const confidenceValue = document.getElementById('confidence-value');
    const progressFill = document.getElementById('progress-fill');
    const resultIcon = document.getElementById('result-icon');
    const dynamicTip = document.getElementById('dynamic-tip');

    // Mappings for UI enhancements based on prediction
    const predictionConfig = {
        'Cyclone': { icon: 'fa-hurricane', tip: 'Seek sturdy shelter immediately. Stay away from windows and exterior doors.' },
        'Earthquake': { icon: 'fa-house-crack', tip: 'Drop, Cover, and Hold On. Move away from glass, outside doors, and walls.' },
        'Flood': { icon: 'fa-water', tip: 'Move to higher ground. Do not walk, swim, or drive through floodwaters.' },
        'Wildfire': { icon: 'fa-fire', tip: 'Evacuate immediately if instructed. Use N95 masks to prevent smoke inhalation.' }
    };

    analyzeBtn.addEventListener('click', async () => {
        // Prepare Form Data
        const formData = new FormData();
        
        if (currentMode === 'upload' && currentFile) {
            formData.append('image', currentFile);
        } else if (currentMode === 'webcam' && base64Capture) {
            formData.append('image_base64', base64Capture);
        } else {
            return; // Safety check
        }

        // UI Transition to Loading
        resultCard.classList.remove('hidden');
        resultContent.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok && data.prediction && !data.error) {
                // Update UI with Results
                const p = data.prediction;
                const conf = data.confidence;
                
                predictionText.textContent = p;
                confidenceValue.textContent = `${conf}%`;
                
                // Animate progress bar with slight delay for effect
                setTimeout(() => {
                    progressFill.style.width = `${conf}%`;
                    
                    // Adjust color based on confidence slightly
                    if(conf < 50) progressFill.style.background = '#ffeb3b';
                    else if(conf < 80) progressFill.style.background = '#ff9800';
                    else progressFill.style.background = '#f44336';
                }, 100);
                
                // Update Icon & Tip
                if (predictionConfig[p]) {
                    resultIcon.className = `fa-solid ${predictionConfig[p].icon}`;
                    dynamicTip.textContent = predictionConfig[p].tip;
                } else {
                    resultIcon.className = 'fa-solid fa-triangle-exclamation';
                    dynamicTip.textContent = 'Please stay alert and safe.';
                }

                // Show Results
                loadingSpinner.classList.add('hidden');
                resultContent.classList.remove('hidden');

            } else {
                throw new Error(data.error || "Prediction failed.");
            }

        } catch (error) {
            console.error("Prediction API Error:", error);
            alert("An error occurred during prediction: " + error.message);
            loadingSpinner.classList.add('hidden');
            resultCard.classList.add('hidden');
        } finally {
            updateAnalyzeButtonState();
        }
    });

});
