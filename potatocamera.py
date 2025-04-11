import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from picamera2 import Picamera2  # Official Raspberry Pi camera library

# Configuration
MODEL_PATH = "/home/lima/tinyinrasp/Model_quant.tflite"
LABELS = ["Healthy", "Early Blight", "Late Blight"]
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Initialize TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
model_height, model_width = input_details[0]['shape'][1:3]

# Initialize Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)},
    transform=cv2.ROTATE_180  # Remove if your camera isn't upside down
)
picam2.configure(config)
picam2.start()

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Preprocess (match your model's exact requirements)
        img = cv2.resize(frame, (model_width, model_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # PiCamera uses BGR by default
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize
        
        # Inference
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        inference_time = (time.time() - start) * 1000
        
        # Get results
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]
        label = f"{LABELS[class_id]}: {confidence:.2f}"
        
        # Display
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{inference_time:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Potato Disease Detector', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()