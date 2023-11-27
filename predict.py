import os

from ultralytics import YOLO
import cv2


video_out = '/Users/ositanwegbu/Documents/GitHub/Manwu_Object_detection/data/video/video_out.mp4'

cap = cv2.VideoCapture("/Users/ositanwegbu/Documents/GitHub/Manwu_Object_detection/data/video/video1.mp4")


ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')


# Load a model
model = YOLO(model_path)  # load a custom model



threshold = 0.5


while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        print(score)
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

results = model("/Users/ositanwegbu/Documents/GitHub/Manwu_Object_detection/data/images/train/IMG_4735.jpg")