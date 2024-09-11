import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model_path = 'yolov5x.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Load the video
video = cv2.VideoCapture("/home/mlsp_student/Desktop/ADAS/VEDIOES/CAR/night.MP4")
output_filename = 'output_video_collision_warning.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

def forward_collision_warning(frame, detections, collision_distance=50):
    height, width, _ = frame.shape
    collision_zone = (0, int(height * 0.3), width, int(height * 0.7))
    frame_with_warning = frame.copy()
    for detection in detections:
        xmin, ymin, xmax, ymax, score, class_id = detection
        if int(class_id) in [2, 3] and score >= 0.3:
            if (xmax > collision_zone[0] and xmin < collision_zone[2] and
                ymax > collision_zone[1] and ymin < collision_zone[3]):
                cv2.rectangle(frame_with_warning, (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)), (0, 0, 255), 2)
                cv2.putText(frame_with_warning, 'Collision Warning!', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame_with_warning

while True:
    success, frame = video.read()
    frame = cv2.resize(frame, (width, height))
    if not success:
        break

    results = model(frame, size=320)
    detections = results.pred[0]

    # Apply forward collision warning
    frame_with_warning = forward_collision_warning(frame, detections)

    videoOut.write(frame_with_warning)
    cv2.imshow("Forward Collision Warning", frame_with_warning)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
videoOut.release()
cv2.destroyAllWindows()

