import cv2
import numpy as np
from openvino.runtime import Core
import time
import requests
import threading

# Weapon detection setup
cls = {0: 'gun', 1: 'heavy-weapon', 2: 'knife'}
core = Core()
model = core.read_model('yolo_11_best_int8.tflite')
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Classification setup
classifier = core.read_model('mode.tflite')
classifier_model = core.compile_model(classifier, "CPU")
input_layer_classifier = classifier_model.input(0)
output_layer_classifier = classifier_model.output(0)

# Camera setup
link = "http://192.168.68.57:4747/video"
cap = cv2.VideoCapture("scene.mp4")  # Replace with your camera source if needed

# Server URL
SERVER_URL = "http://192.168.68.65:5000/update_frame"

# Shared variables
latest_frame = None
frame_lock = threading.Lock()
running = True
notification = False
frame_to_classify = None
time_for_clock = False

def classify_loop():
    global notification, running, frame_to_classify
    while running:
        if frame_to_classify is not None:
            resized_frame = cv2.resize(frame_to_classify, (224, 224))
            input_data = np.expand_dims(resized_frame, 0).astype(np.float32) / 255.0
            results = classifier_model([input_data])[output_layer_classifier]
            m = np.argmax(results)
            notification = results[0][m] > 0.1
            frame_to_classify = None
        time.sleep(0.01)  # Reduced sleep time for faster response

def action_on_timeout():
    global notification, time_for_clock
    notification = False
    time_for_clock = False

def stopwatch(duration_in_minutes=10):
    global time_for_clock
    time_for_clock = True
    duration_in_seconds = duration_in_minutes * 60
    time.sleep(duration_in_seconds)
    action_on_timeout()

def process_frame(frame):
    global frame_to_classify, notification, time_for_clock
    frame_height, frame_width = frame.shape[:-1]
    input_shape = input_layer.shape
    resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(resized_frame, 0).astype(np.float32) / 255.0

    results = compiled_model([input_data])[output_layer]
    outputs = np.array([cv2.transpose(results[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        conf = scores[index]
        id_ = class_ids[index]
        if conf > 0.2:
            x1 = int(box[0] * frame_width)
            y1 = int(box[1] * frame_height)
            x2 = int((box[0] + box[2]) * frame_width)
            y2 = int((box[1] + box[3]) * frame_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {cls[int(id_)]}, Conf: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if frame_to_classify is None and not notification:
                frame_to_classify = frame[y1:y2, x1:x2]
            elif notification:
                frame_to_classify = None
            if notification and not time_for_clock:
                threading.Thread(target=stopwatch).start()

    # Add notification status to the frame
    notification_status = "Active" if notification else "Inactive"
    cv2.putText(frame, f"Notification: {notification_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

def send_frame_thread():
    global latest_frame, running
    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)  # Short sleep if no frame is available
                continue
            frame_to_send = latest_frame.copy()
        
        _, img_encoded = cv2.imencode('.jpg', frame_to_send)
        try:
            response = requests.post(SERVER_URL, 
                                     files={'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')},
                                     timeout=1)
            if response.status_code != 200:
                print(f"Failed to send frame. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending frame: {e}")
        
        time.sleep(0.03)  # Adjust this delay to control send rate

# Start the classification thread
classify_thread = threading.Thread(target=classify_loop)
classify_thread.start()

# Start the sending thread
sending_thread = threading.Thread(target=send_frame_thread)
sending_thread.start()

fps_start_time = time.time()
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_counter += 1
        frame = cv2.flip(frame, 1)
        
        processed_frame = process_frame(frame)
        
        with frame_lock:
            latest_frame = processed_frame
        
        fps_end_time = time.time()	
        time_diff = fps_end_time - fps_start_time
        fps = frame_counter / time_diff
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
       # cv2.imshow("Object Detection", processed_frame)
        
       # if cv2.waitKey(1) & 0xFF == ord('q'):
       #     break

finally:
    # Cleanup
    running = False
    cap.release()
    cv2.destroyAllWindows()
    classify_thread.join()  # Wait for the classification thread to finish
    sending_thread.join()  # Wait for the sending thread to finish

print("Program ended")
