import cv2

video = cv2.VideoCapture("/home/mlsp_student/Desktop/ADAS/VEDIOES/CAR/night.MP4")
#/home/mlsp_student/Desktop/ADAS/VEDIOES/CAB/V_000489.mov
#/home/mlsp_student/Desktop/ADAS/VEDIOES/CAR/night.MP4
frame_rate = video.get(cv2.CAP_PROP_FPS)
print(f"Frame Rate: {frame_rate} fps")
video.release()

