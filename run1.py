import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from PIL import Image


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]


cap = cv2.VideoCapture(0)


class_names = ['bad', 'good', 'mediocre']

while True:

    ret, frame = cap.read()
    if not ret:
        break


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = Image.fromarray(img)
    img = np.expand_dims(img, axis=0)
    img = (np.float32(img) - 127.5) / 127.5  # Chuẩn hóa [-1, 1]


    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])


    prediction = np.squeeze(output_data)
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    predicted_class = class_names[class_index]


    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Orange Quality Classification', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()