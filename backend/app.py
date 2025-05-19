import logging
logger = logging.getLogger("SignServer")
logging.basicConfig(level=logging.DEBUG)

# Replace all print() statements with:
logger.debug("message")
logger.info("message")
logger.error("message")


import os
os.environ['FLASK_ENV'] = 'development'

from flask import Flask, json, jsonify, redirect, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
import cv2
import base64
import mediapipe as mp
from collections import deque

mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'

CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=25, ping_timeout=60)

Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Layer = tf.keras.layers.Layer
Lambda = tf.keras.layers.Lambda

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = Dense(input_shape[-1], activation='softmax')
        self.multiply = Multiply()
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_weights = self.attention_weights(inputs)
        weighted = self.multiply([inputs, attention_weights])
        return tf.reduce_sum(weighted, axis=1)

# Define labels as used during training
labels = ["FINE1", "DEAF1", "SOON1", "DOCTOR1", "WHAT2", "RAIN", "NAME", "MOTHER", 
    "DRESS", "HEARING", "HELLO", "5DOLLARS", "9OCLOCK", "SICK", "CHILD", "THREE", 
    "SMART", "EVERYNIGHT", "HELP", "ONE", "TUESDAY", "FROM", "WHAT1", "GIRAFFE", 
    "ALLIGATOR", "FRECKLES", "CURLYHAIR", "WASHDISHES", "POPULAR", "PHYSICS", 
    "8HOUR", "CLEAN", "TEACHER", "BUG", "HAMBURGER", "EMAIL", "APRIL", "OLD", 
    "SOFTBALL", "GOODBYE", "NURSE", "PRETTY", "ROOM", "FAMILY", "LASTWEEK", 
    "BOYFRIEND", "WEATHER", "ISLAND", "HURT", "FRIEND", "SHAMPOO", "DOCTOR2", 
    "DEAF2", "SOON2", "FINE2"]  

# Load the model
def load_model(): 
    global model
    model = tf.keras.models.load_model('lstm_model81.h5', custom_objects={'AttentionLayer': AttentionLayer})
    print("Model loaded successfully!")

# @socketio.on("predict")
# def handle_prediction(data):
#     print("Received prediction request")
#     try:
#         frames = np.array(data["frames"], dtype=np.float32)
#         if frames.shape != (60, 126):
#             emit("prediction", {"error": "Invalid input shape"})
#             return
#         frames = np.expand_dims(frames, axis=0)  # Shape: (1, 60, 126)
#         prediction = model.predict(frames)[0]
#         label = labels[np.argmax(prediction)]
#         confidence = float(np.max(prediction))
#         emit("prediction", {"label": label, "confidence": confidence})
#     except Exception as e:
#         emit("prediction", {"error": str(e)})

@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/test_camera")
# def test_camera():
#     return render_template("test_camera.html")

# Keep the HTTP endpoint for compatibility
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get keypoints data from the request
        keypoints_data = request.json.get('keypoints', [])
        
        # Process data and make prediction
        result = process_prediction(keypoints_data)
        
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result)

# Process prediction data and return results
def process_prediction(keypoints_data):
    try:
        # Convert to numpy array
        sequence = np.array(keypoints_data)
        
        # Ensure we have the right shape for prediction
        if sequence.shape != (60, 126):
            return {'error': 'Expected input shape (60, 126)'}
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)  # Shape: (1, 60, 126)
        res = model.predict(sequence_batch)[0]

        # Debug: print raw model output
        print("Model raw output:", res.tolist())
        
        # Get the predicted action
        predicted_action = labels[np.argmax(res)]
        confidence = float(res[np.argmax(res)])
        
        # Get top predictions
        top_indices = np.argsort(res)[::-1][:5]  # Get top 5 predictions
        top_actions = [labels[i] for i in top_indices]
        top_probabilities = [float(res[i]) for i in top_indices]
        
        return {
            'prediction': predicted_action,
            'confidence': confidence,
            'top_predictions': [{'action': action, 'probability': prob} 
                                for action, prob in zip(top_actions, top_probabilities)]
        }
    except Exception as e:
        return {'error': str(e)}
    
@socketio.on('frame')
def handle_frame(data):
    logger.info(f"Received frame. Queue size: {len(frame_queue)}")
    
    b64image = data.get('image')
    if not b64image:
        return

    jpg_bytes = base64.b64decode(b64image)
    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb_frame)

    # Draw landmarks for OpenCV preview
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


    # Prepare landmark data to send to frontend
    def normalize_landmarks(landmarks):
        if not landmarks:
            return []
        return [{'x': lm.x, 'y': lm.y} for lm in landmarks.landmark]

    landmarks = {
        'left_hand': normalize_landmarks(results.left_hand_landmarks),
        'right_hand': normalize_landmarks(results.right_hand_landmarks),
    }

    # # Send annotated image
    # _, buffer = cv2.imencode('.jpg', frame)
    # annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    # emit('annotated_frame', {
    #     'image': annotated_b64,
    #     'landmarks': landmarks,
    # })

    # Queue keypoints for prediction
    keypoints = extract_keypoints(results)
    frame_queue.append(keypoints)

    if len(frame_queue) == 60:
        logger.info("Got 60 frames. Predicting...")
        result = process_prediction(list(frame_queue))
        emit('prediction_result', result, broadcast=False)
        logger.info(f"Prediction emitted: {result}")
        frame_queue.clear()


# WebSocket endpoint for predictions
@socketio.on('predict_sign')
def handle_prediction(data):
    print("[DEBUG] Received predict_sign event with", len(data.get('keypoints', [])), "frames")
    print("Received predict_sign event:", data)
    try:
        # Get keypoints data from the WebSocket message
        keypoints_data = data.get('keypoints', [])
        
        # Process prediction
        result = process_prediction(keypoints_data)
        
        if 'error' in result:
            emit('prediction_error', {'error': result['error']})
            return
        
        # Send prediction back to all clients (including the sender)
        print("Sending prediction_result:", result)
        emit('prediction_result', result, broadcast=False)
        
    except Exception as e:
        emit('prediction_error', {'error': str(e)})

# Holistic setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
frame_queue = deque(maxlen=60)


def extract_keypoints(results):
    def get_landmarks(landmarks, count):
        if landmarks:
            coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            return np.array(coords).flatten()
        return np.zeros(count * 3)

    left_hand = get_landmarks(results.left_hand_landmarks, 21)
    right_hand = get_landmarks(results.right_hand_landmarks, 21)
    logger.debug("Left hand landmarks: %s", results.left_hand_landmarks)
    logger.debug("Right hand landmarks: %s", results.right_hand_landmarks)

    return np.concatenate([left_hand, right_hand])


# WebSocket connection event
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to server'}, broadcast=False)

# WebSocket disconnection event
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
    load_model()
    logger.info("Starting Sign Language Recognition Server...")
    socketio.run(app, host="0.0.0.0", port=5070)
