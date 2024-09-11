import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model_path = 'yolov5x.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Load the video
video = cv2.VideoCapture("/home/mlsp_student/Desktop/ADAS/VEDIOES/CAB/V_000489.mov")
#home/mlsp_student/Desktop/ADAS/VEDIOES/CAB/V_000489.mov
#/home/mlsp_student/Desktop/ADAS/VEDIOES/CAR/night.MP4
output_filename = 'output_video_headlights.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

def adaptive_headlight(frame, detections, brightness_factor=1.5):
    frame_with_headlights = frame.copy()
    # Convert detections tensor to numpy array
    detections_array = detections.cpu().numpy() if torch.cuda.is_available() else detections.numpy()
    
    # Determine headlight status
    headlight_status = "Full"
    if len(detections_array) > 0:
        headlight_status = "Dim"
        hsv = cv2.cvtColor(frame_with_headlights, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * brightness_factor, 0, 255)
        frame_with_headlights = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay headlight status text on the frame
    cv2.putText(frame_with_headlights, f"Headlights: {headlight_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame_with_headlights

while True:
    success, frame = video.read()
    frame = cv2.resize(frame, (width, height))
    if not success:
        break

    results = model(frame, size=320)
    detections = results.pred[0]

    # Apply adaptive headlights and show status
    frame_with_headlights = adaptive_headlight(frame, detections)

    videoOut.write(frame_with_headlights)
    cv2.imshow("Adaptive Headlights", frame_with_headlights)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
videoOut.release()
cv2.destroyAllWindows()

