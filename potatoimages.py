import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

# Configuration
MODEL_PATH = "/home/lima/tinyinrasp/Model_quant.tflite"
IMAGE_PATH = "/home/lima/tinyinrasp/images/image1.jpg"  # Update extension if different
LABELS = ["Healthy", "Early Blight", "Late Blight"]

# Initialize TFLite interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

# Read and preprocess the image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Failed to load image at {IMAGE_PATH}")
    exit()

img = cv2.resize(image, (width, height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0).astype(np.uint8)

# Run inference
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

# Get result
class_id = np.argmax(predictions)
confidence = predictions[0][class_id]
label = f"{LABELS[class_id]}: {confidence:.2f}"

# Display result
cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Prediction', image)
print(f"Prediction: {label}")
print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")

cv2.waitKey(0)
cv2.destroyAllWindows()
