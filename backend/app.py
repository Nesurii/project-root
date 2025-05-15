from flask import Flask, render_template, request, jsonify 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import CORS
from tensorflow.keras.layers import Dense, Multiply, Layer
from tensorflow.keras.layers import Lambda

import os
#ngrok http --url=classic-proven-kingfish.ngrok-free.app 80
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Set up CORS for regular HTTP routes
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

# Set up SocketIO with CORS
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create a custom AttentionLayer exactly as in your original code
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

# Load your actions array
actions = np.array([
    "hello", "fine", "goodbye", "name", "friend", "one", "sick", "help", "deaf", "hearing",
    "what", "from", "email", "weather", "rain", "three", "tuesday", "april", "child", "old",
    "9oclock", "5dollars", "8hours", "soon", "lastweek", "family",  "mother", "boyfriend", 
    "smart", "pretty","room", "teacher", "physics", "popular", "bug", "softball", "everynight", 
    "shampoo", "washdishes", "clean", "freckles", "curlyhair", "dress", "doctor", "hurt", "island",
    "nurse", "hamburger", "crocodile", "giraffe"
])

# Load model
model = None

def load_lstm_model():
    global model
    model = load_model('lstm_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
    print("Model loaded successfully!")

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
        if sequence.shape[0] != 60:
            return {'error': 'Need exactly 60 frames of keypoints'}
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)
        res = model.predict(sequence_batch)[0]
        
        # Get the predicted action
        predicted_action = actions[np.argmax(res)]
        confidence = float(res[np.argmax(res)])
        
        # Get top predictions
        top_indices = np.argsort(res)[::-1][:5]  # Get top 5 predictions
        top_actions = [actions[i] for i in top_indices]
        top_probabilities = [float(res[i]) for i in top_indices]
        
        return {
            'prediction': predicted_action,
            'confidence': confidence,
            'top_predictions': [{'action': action, 'probability': prob} 
                              for action, prob in zip(top_actions, top_probabilities)]
        }
    except Exception as e:
        return {'error': str(e)}

# WebSocket endpoint for predictions
@socketio.on('predict_sign')
def handle_prediction(data):
    try:
        # Get keypoints data from the WebSocket message
        keypoints_data = data.get('keypoints', [])
        
        # Process prediction
        result = process_prediction(keypoints_data)
        
        if 'error' in result:
            emit('prediction_error', {'error': result['error']})
            return
        
        # Send prediction back to all clients (including the sender)
        emit('prediction_result', result, broadcast=True)
        
    except Exception as e:
        emit('prediction_error', {'error': str(e)})

# WebSocket connection event
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to server'})

# WebSocket disconnection event
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize model
    load_lstm_model()
    
    # Run Flask app with SocketIO - bind to all interfaces
    # Make sure to use 0.0.0.0 to accept connections from any IP
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


