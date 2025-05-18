from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'

CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

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


@app.route("/test_camera")
def test_camera():
    return render_template("test_camera.html")

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


if __name__ == "__main__":
    load_model()
    socketio.run(app, host="0.0.0.0", port=5080)
