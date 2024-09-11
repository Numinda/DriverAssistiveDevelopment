import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model_path = 'yolov5x.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Load the video
video = cv2.VideoCapture("/home/mlsp_student/Desktop/ADAS/VEDIOES/CAR/collision.MP4")
output_filename = 'output_video2.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

# Set the positions of the horizontal lines
line_y1 = int(height * 0.6)  # Position of the first line (e.g., 60% of the height)
line_y2 = int(height * 0.8)  # Position of the second line (e.g., 80% of the height)

#Set the length of the horizontal lines
line_x_start = int(width * 0.3)  # Start at 30% of the width
line_x_end = int(width * 0.7)    # End at 70% of the width

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Resize the frame for processing
    frame_resized = cv2.resize(frame, (width, height))

    # Perform inference on the frame
    results = model(frame_resized)

    # Draw the horizontal lines on the frame
    cv2.line(frame_resized, (line_x_start, line_y1), (line_x_end, line_y1), (0, 255, 0), 2)
    cv2.line(frame_resized, (line_x_start, line_y2), (line_x_end, line_y2), (0, 255, 0), 2)

    # Process detections
    for detection in results.xyxy[0]:  # Extract detections
        x1, y1, x2, y2, conf, cls = detection

        # Check if the object falls between the two lines
        if y1 > line_y1 and y2 < line_y2:
            # Object is between the two lines, so we alert the driver
            cv2.putText(frame_resized, 'Warning: Forward Collision!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red rectangle for warning

    # Save the processed frame to the output video
    videoOut.write(frame_resized)

    # Display the frame (optional, can be removed in final implementation)
    cv2.imshow('Frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
videoOut.release()
cv2.destroyAllWindows()

