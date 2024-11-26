// Global variables
const video = document.getElementById('video');
let isCollectingData = false;
let personName = '';
let dataPoints = [];
const CAPTURE_INTERVAL = 500; // Capture every 500ms
const REQUIRED_SAMPLES = 30; // Number of samples to collect per person

// Load all required face-api models
async function loadModels() {
    await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
    ]);
    console.log('Models loaded successfully');
    startVideo();
}

// Start video stream
function startVideo() {
    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
            initializeFaceDetection();
        });
}

// Initialize face detection and canvas
function initializeFaceDetection() {
    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);
    
    startFaceDetection(canvas, displaySize);
}

// Main face detection loop
async function startFaceDetection(canvas, displaySize) {
    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(
            video, 
            new faceapi.TinyFaceDetectorOptions()
        )
        .withFaceLandmarks()
        .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw face detections
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

        // Collect data if in collection mode
        if (isCollectingData && detections.length > 0) {
            collectFaceData(detections[0]);
        }
    }, 100);
}

// Start data collection for a person
function startDataCollection(name) {
    personName = name;
    isCollectingData = true;
    dataPoints = [];
    console.log(`Started collecting data for ${personName}`);
}

// Collect face data
async function collectFaceData(detection) {
    if (dataPoints.length >= REQUIRED_SAMPLES) {
        isCollectingData = false;
        saveFaceData();
        return;
    }

    const faceData = {
        landmarks: detection.landmarks.positions,
        descriptor: detection.descriptor,
        timestamp: Date.now()
    };

    dataPoints.push(faceData);
    console.log(`Collected sample ${dataPoints.length}/${REQUIRED_SAMPLES}`);
}

// Save collected face data
async function saveFaceData() {
    const faceData = {
        name: personName,
        samples: dataPoints,
        dateCollected: new Date().toISOString()
    };

    // Save to IndexedDB or localStorage
    localStorage.setItem(`faceData_${personName}`, JSON.stringify(faceData));
    console.log(`Saved face data for ${personName}`);
}

// Train model with collected data
async function trainModel() {
    const labeledFaceDescriptors = [];
    
    // Get all stored face data
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith('faceData_')) {
            const faceData = JSON.parse(localStorage.getItem(key));
            const descriptors = faceData.samples.map(sample => 
                new Float32Array(Object.values(sample.descriptor))
            );
            
            labeledFaceDescriptors.push(
                new faceapi.LabeledFaceDescriptors(faceData.name, descriptors)
            );
        }
    }

    return new faceapi.FaceMatcher(labeledFaceDescriptors);
}

// Start face recognition with trained model
async function startFaceRecognition() {
    const faceMatcher = await trainModel();
    console.log('Face recognition model trained and ready');
    
    // Update detection loop to include recognition
    const canvas = document.querySelector('canvas');
    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video)
            .withFaceLandmarks()
            .withFaceDescriptors();
            
        const results = detections.map(detection => {
            return faceMatcher.findBestMatch(detection.descriptor);
        });
        
        results.forEach((result, i) => {
            const box = detections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { 
                label: result.toString(),
                lineWidth: 2 
            });
            drawBox.draw(canvas);
        });
    }, 100);
}

// Usage example commands:
// loadModels(); // Call this first
// startDataCollection('John'); // Start collecting data for John
// After collection completes automatically:
// startFaceRecognition(); // Start recognition with trained model
