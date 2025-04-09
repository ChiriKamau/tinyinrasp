import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

# Configuration
MODEL_PATH = "/home/lima/tinyinrasp/Model_quant.tflite"  # Path to your quantized model
LABELS = ["Healthy", "Early Blight", "Late Blight"]  # Update with your classes
INPUT_SIZE = (224, 224)  # Should match your model's input shape

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]  # Get model input shape

# Camera setup (use 0 for default camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS counter
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = np.expand_dims(img, axis=0).astype(np.uint8)  # Add batch dimension
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Get results
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    label = f"{LABELS[class_id]}: {confidence:.2f}"
    
    # Display results
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Potato Disease Detector', frame)
    
    # FPS calculation
    frame_count += 1
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()