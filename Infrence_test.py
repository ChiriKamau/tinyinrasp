import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Path to your model
MODEL_PATH = "/home/lima/tinyinrasp/model_quantized.tflite"  # Update this to your model path

# Load and allocate the model
print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model information
print("Model loaded successfully!")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output type: {output_details[0]['dtype']}")

# Create a sample input (replace this with actual data preprocessing)
input_shape = input_details[0]['shape']
sample_input = np.random.random(input_shape).astype(input_details[0]['dtype'])

# Run inference
print("\nRunning inference...")
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

print(f"Inference completed in {(end_time - start_time)*1000:.2f} ms")
print(f"Output shape: {output.shape}")
print(f"Sample output values: {output.flatten()[:5]}...")  # Show first 5 values