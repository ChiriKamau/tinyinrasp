import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("your_model.h5")

# Convert to TFLite with quantization (reduces size + speeds up inference)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize weights
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # 8-bit
converter.inference_input_type = tf.uint8  # Quantize input/output

tflite_model = converter.convert()

# Save the quantized model
with open("model_quant.tflite", "wb") as f:
    f.write(tflite_model)