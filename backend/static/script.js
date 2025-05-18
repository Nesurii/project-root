// Select DOM elements
const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const resultDiv = document.getElementById("result");

// State variables
let showLandmarks = false;
let isRunning = false;
let frameBuffer = [];

// Initialize Socket.IO client
const socket = io("https://b985-103-200-33-5.ngrok-free.app/");

// UI controls
document.getElementById("start").onclick = () => {
  isRunning = true;
  frameBuffer = [];
  resultDiv.textContent = "Prediction: ...";
};

document.getElementById("stop").onclick = () => {
  isRunning = false;
  resultDiv.textContent = "Prediction: Stopped";
};

document.getElementById("toggle-landmarks").onchange = (e) => {
  showLandmarks = e.target.checked;
};

// MediaPipe Holistic setup
const holistic = new Holistic({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
holistic.onResults(onResults);

// Start camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({image: videoElement});
  },
  width: 640,
  height: 480
});
camera.start();

// Extract and flatten 126 keypoints (pose + left hand + right hand)
function extractKeypoints(results) {
  const lm = (list) =>
    list ? list.map(p => [p.x, p.y, p.z].map(v => v ?? 0)).flat() : Array(21 * 3).fill(0);

  const pose = results.poseLandmarks ? results.poseLandmarks.map(p => [p.x, p.y, p.z]).flat() : Array(33 * 3).fill(0);
  const lh = lm(results.leftHandLandmarks);
  const rh = lm(results.rightHandLandmarks);

  return [...pose, ...lh, ...rh];  // 33*3 + 21*3 + 21*3 = 126
}

// Called on every MediaPipe results event
function onResults(results) {
  if (!isRunning) return;

  // Draw the video frame and landmarks if enabled
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (showLandmarks) {
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
    drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#00FF00', radius: 4 });

    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#FF0000', lineWidth: 2 });
    drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#FF0000', radius: 2 });

    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#0000FF', lineWidth: 2 });
    drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#0000FF', radius: 1 });
  }

  // Extract keypoints and buffer frames
  const keypoints = extractKeypoints(results);
  frameBuffer.push(keypoints);

  // Send batch of 60 frames to backend via Socket.IO
  if (frameBuffer.length === 60) {
    console.log("[DEBUG] Emitting predict_sign with batch size:", frameBuffer.length);
    socket.emit("predict_sign", { keypoints: frameBuffer });
    frameBuffer = [];
  }

  canvasCtx.restore();
}

// Socket.IO event listeners

socket.on("connect", () => {
  console.log("Socket.IO connected");
  resultDiv.textContent = "Socket.IO connected, ready to predict.";
});

socket.on("disconnect", () => {
  console.log("Socket.IO disconnected");
  resultDiv.textContent = "Socket.IO disconnected.";
});

socket.on("connect_error", (err) => {
  console.error("Socket.IO connection error:", err);
  resultDiv.textContent = "Connection error: " + err.message;
});

socket.on("prediction_result", (data) => {
  // Display the prediction label and confidence
  resultDiv.textContent = `Prediction: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
   // Send prediction to Flutter app via flutter_inappwebview handler
  if (window.flutter_inappwebview) {
    window.flutter_inappwebview.callHandler('sendPredictionToFlutter', data);
  }
});

socket.on("prediction_error", (data) => {
  console.error("Prediction error:", data.error);
  resultDiv.textContent = "Error: " + data.error;
});
