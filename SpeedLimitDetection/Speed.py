#/home/mlsp_student/Desktop/ADAS/VEDIOES/speed.mp4
import cv2
import torch
import pytesseract
import numpy as np

# Load the YOLOv5 model
model_path = 'best_93.pt'  # Update this to your model path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Load the video
video_path = '/home/mlsp_student/Desktop/ADAS/VEDIOES/speed.mp4'  # Update this to your video file path
video = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not video.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
    exit()

# Define a function to extract speed limit from detected signs
def extract_speed_limit(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    text = pytesseract.image_to_string(roi, config='--psm 6')
    try:
        return int(text.strip())
    except ValueError:
        return 0

# Constant vehicle speed (35 km/h)
vehicle_speed = 35

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video or unable to read frame.")
        break

    # Resize the frame for processing
    frame_resized = cv2.resize(frame, (1280, 720))

    # Perform inference on the frame
    results = model(frame_resized)

    # Process detections
    speed_limit = 0
    for detection in results.xyxy[0]:  # Extract detections
        x1, y1, x2, y2, conf, cls = detection

        # Debug: print detection details
        print(f"Detection: {x1}, {y1}, {x2}, {y2}, Confidence: {conf}, Class: {cls}")

        # Check if the object is a speed limit sign (assuming class 0 is for speed limit signs)
        if int(cls) == 0:
            speed_limit = extract_speed_limit(frame_resized, (x1, y1, x2, y2))
            if speed_limit > 0:
                cv2.putText(frame_resized, f'Speed Limit: {speed_limit} km/h', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Speed Limit Detected: {speed_limit} km/h")  # Debug: print detected speed limit

    # Compare the constant vehicle speed with the speed limit and display warning if needed
    if speed_limit > 0:
        if vehicle_speed > speed_limit:
            cv2.putText(frame_resized, f'Speed {vehicle_speed} km/h exceeds limit!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Warning: Speed {vehicle_speed} km/h exceeds limit!")  # Debug: print warning message
        else:
            cv2.putText(frame_resized, f'Speed {vehicle_speed} km/h is within limit.', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Speed {vehicle_speed} km/h is within limit.")  # Debug: print within limit message

    # Display the frame
    cv2.imshow('Speed Warning', frame_resized)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
