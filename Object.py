import cv2
import numpy as np
from openvino.runtime import Core
import time


cls = {0:'gun',1:'heavy-weapon',2:'knife'}
classes={0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

core = Core()


model = core.read_model('11s.tflite')
compiled_model = core.compile_model(model, "CPU")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


#Camera
link = "http://192.168.68.57:4747/"
cap = cv2.VideoCapture(link)


fps_start_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()
    frame_height,frame_width = frame.shape[:-1]
    if not ret:
        print("Failed to grab frame")
        break
    frame_counter += 1
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = face_classifier.detectMultiScale(gray,1.3,5)
# if len(faces)>0:
    input_shape = input_layer.shape
    resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(resized_frame, 0).astype(np.float32) / 255.0

    # Perform inference
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
        if conf>0.2:
            x1 = int(box[0]*frame_width)
            y1 = int(box[1]*frame_height)
            x2 = int((box[0]+box[2])*frame_width)
            y2 = int((box[1]+box[3])*frame_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {cls[int(id_)]}, Conf: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # detect(frame,faces)
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = frame_counter / time_diff
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
