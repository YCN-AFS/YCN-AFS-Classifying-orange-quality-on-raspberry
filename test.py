import cv2
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from class_yolo import YOLOv8
import tensorflow as tf


np.set_printoptions(suppress=True)

# Load YOLO model using YOLOv8 class
yolo_model_path = 'new_orange.onnx'
yolov8_detector = YOLOv8(yolo_model_path, conf_thres=0.7, iou_thres=0.5)

# Load Keras model
# keras_model = load_model("model3_class/keras_model.h5", compile=False)
keras_model = tf.lite.Interpreter(model_path="model.tflite")
keras_model.allocate_tensors()

# Load class names
class_names = open("model3_class/labels.txt", "r").readlines()

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# Define color map for classes
colors = np.random.randint(0, 255, size=(len(class_names), 3))

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

def predict_image(image_array):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image_array
    prediction = keras_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Initialize dictionary to store classification results
classification_results = {}

def update_classification_results(track_id, class_name):
    if track_id not in classification_results:
        classification_results[track_id] = {
            '0 good': 0,
            '1 bad': 0,
            '2 mediocre': 0,
            'start_time': time.time(),
            'final_class': None
        }

    if class_name == "0 good":
        classification_results[track_id]['0 good'] += 1
    elif class_name == "1 bad":
        classification_results[track_id]['1 bad'] += 1
    elif class_name == "2 mediocre":
        classification_results[track_id]['2 mediocre'] += 1

def determine_final_class(track_id):
    results = classification_results[track_id]
    total_classifications = results['0 good'] + results['1 bad'] + results['2 mediocre']

    if total_classifications == 0:
        return None

    good_ratio = results['0 good'] / total_classifications
    bad_ratio = results['1 bad'] / total_classifications
    mediocre_ratio = results['2 mediocre'] / total_classifications

    if good_ratio >= 0.7:
        return "Good"
    elif bad_ratio >= 0.3:
        return "Bad"
    elif mediocre_ratio >= 0.3:
        return "Mediocre"
    else:
        return None

def detect_and_classify():
    cap = cv2.VideoCapture(0)
    tracked_oranges = {}

    try:
        while cap.isOpened():
            start_time = time.time()  # Start time for FPS calculation

            ret, image = cap.read()
            if not ret:
                break

            original_size = image.shape[1], image.shape[0]  # Save the original size

            # Resize the image for YOLO processing
            resized_image = cv2.resize(image, (640, 640))  # Maintain the size to match the model's expected input size

            # Run the YOLOv8 model
            boxes, scores, class_ids = yolov8_detector(resized_image)

            detections = []
            for box, confidence, class_id in zip(boxes, scores, class_ids):
                if confidence > 0.6:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, 0])  # Class ID set to 0 for all oranges

            # Update tracks using DeepSort
            tracks = tracker.update_tracks(detections, frame=resized_image)

            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    cropped_frame = resized_image[y1:y2, x1:x2]
                    if cropped_frame.size > 0:
                        cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                    else:
                        continue

                    # Ensure track_id is in classification_results
                    if track_id not in classification_results:
                        classification_results[track_id] = {
                            '0 good': 0,
                            '1 bad': 0,
                            '2 mediocre': 0,
                            'start_time': time.time(),
                            'final_class': None
                        }

                    # Preprocess the image and classify
                    image_array = preprocess_image(cropped_image)
                    predicted_label, confidence_score = predict_image(image_array)

                    update_classification_results(track_id, predicted_label)

                    # Determine the final class if 5 seconds have passed since the first detection
                    if time.time() - classification_results[track_id]['start_time'] > 5 and \
                            classification_results[track_id]['final_class'] is None:
                        final_class = determine_final_class(track_id)
                        classification_results[track_id]['final_class'] = final_class
                        if final_class:
                            tracked_oranges[track_id] = final_class

                    final_class = classification_results[track_id]['final_class']

                    if final_class:
                        if final_class == "Good":
                            color = (0, 255, 0)
                        elif final_class == "Bad":
                            color = (0, 0, 255)
                        elif final_class == "Mediocre":
                            color = (255, 255, 0)
                    else:
                        color = (255, 255, 255)  # White for undetermined class

                    # Display the classification result or placeholder on the resized image (640x640)
                    cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
                    display_text = f"ID: {track_id} {final_class if final_class else 'Classifying...'}"
                    cv2.putText(resized_image, display_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # Display FPS on the resized image (640x640)
            cv2.putText(resized_image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize the image back to its original size
            final_image = cv2.resize(resized_image, original_size)
            cv2.imshow("Detect and classification orange", final_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print the summary
        total_oranges = len(tracked_oranges)
        good_oranges = sum(1 for v in tracked_oranges.values() if v == "Good")
        bad_oranges = sum(1 for v in tracked_oranges.values() if v == "Bad")
        mediocre_oranges = sum(1 for v in tracked_oranges.values() if v == "Mediocre")

        print(f"Total oranges: {total_oranges}")
        print(f"Good oranges: {good_oranges}")
        print(f"Bad oranges: {bad_oranges}")
        print(f"Mediocre oranges: {mediocre_oranges}")

if __name__ == "__main__":
    detect_and_classify()
