import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Path to your model
MODEL_PATH = "/home/lima/tinyinrasp/Model_quantized.tflite"  # Update this to your model path

# Debugging: Check if the model path exists
import os
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file does not exist at {MODEL_PATH}")
else:
    print(f"Model found at {MODEL_PATH}")

# Load and allocate the model
print("Loading model...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Get input and output details
try:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debugging: Print model information
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
except Exception as e:
    print(f"Error getting input/output details: {e}")

# Create a sample input (replace this with actual data preprocessing)
input_shape = input_details[0]['shape']
print(f"Sample input shape: {input_shape}")
sample_input = np.random.random(input_shape).astype(input_details[0]['dtype'])

# Debugging: Print the sample input
print(f"Sample input values: {sample_input.flatten()[:5]}...")

# Run inference
print("\nRunning inference...")
start_time = time.time()
try:
    interpreter.set_tensor(input_details[0]['index'], sample_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print("Inference completed successfully.")
except Exception as e:
    print(f"Error during inference: {e}")

end_time = time.time()

# Debugging: Print inference time and output
print(f"Inference completed in {(end_time - start_time)*1000:.2f} ms")
try:
    print(f"Output shape: {output.shape}")
    print(f"Sample output values: {output.flatten()[:5]}...")  # Show first 5 values
except Exception as e:
    print(f"Error accessing output: {e}")
