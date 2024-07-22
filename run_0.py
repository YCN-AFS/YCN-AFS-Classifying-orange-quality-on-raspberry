import numpy as np
import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


labels = ['good', 'bad', 'mediocre']


def run_inference(input_data):
    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)

    return frame_batch



import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    results = run_inference(processed_frame)
    predicted_class = np.argmax(results)
    predicted_label = labels[predicted_class]
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Realtime Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()