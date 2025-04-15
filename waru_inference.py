import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "/home/lima/tinyinrasp/Model_quant.tflite"
IMAGE_PATH = "/home/lima/tinyinrasp/images/image1.JPG"
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

# Resize and preprocess
img = cv2.resize(image, (width, height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Float32 normalization

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

# Display result with matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for matplotlib
plt.imshow(image)
plt.title(label)
plt.axis('off')  # Turn off axes
plt.show()

print(f"Prediction: {label}")
print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
