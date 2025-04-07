import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path='/home/pi/tinyinrasp/Model_quantized.tflite')
interpreter.allocate_tensors()

# Load labels
with open('/home/pi/tflite_model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(image, axis=0)
    return np.array(input_data, dtype=np.float32)

# Run inference on an image
input_data = preprocess_image('/home/pi/tflite_model/test_image.jpg')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get and print the output
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data[0])
print(f'Predicted class: {labels[predicted_class]}')
