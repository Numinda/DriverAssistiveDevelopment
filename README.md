Driver Assistance System with YOLOv5 and Jetson
This project develops a real-time driver assistance system deployed on an NVIDIA Jetson board. It leverages the YOLOv5 deep learning model for object detection, enhancing driver awareness and safety on the road.

Key functionalities include:

Speed Limit Recognition and Assist: Accurately detects speed limit signs and alerts drivers when exceeding the posted limit.
Forward Collision Warning: Continuously monitors the environment, identifying potential collisions and providing timely warnings to prevent accidents.
Adaptive Headlight Control: Automatically adjusts headlights based on ambient light conditions and detected vehicles, improving visibility for both the driver and oncoming traffic.
Features:
Real-time object detection using YOLOv5: Efficiently detects relevant objects like vehicles, pedestrians, and traffic signs.
Bird's Eye View (BEV) visualization (Simulated): Offers a top-down perspective of the environment for enhanced situational awareness (Note: Currently implemented in a simulated environment).
Customizable confidence threshold: Tailor the system to prioritize detections with a higher degree of certainty.
Class filtering: Focus on specific object categories for tailored alerts and actions.
Prerequisites:
NVIDIA Jetson board
Python 3.x
OpenCV
PyTorch
NumPy
Installation:
Clone this repository.
Install the required dependencies:
Bash
pip3 install torch opencv numpy   

Use code with caution.

Note: Additional libraries or setup might be required for Jetson deployment. Refer to the project documentation for specific instructions.

Usage:
Download pre-trained YOLOv5 weights or train your own model for the Jetson environment.
Configure the code for deployment on the Jetson board.
Run the script to activate the driver assistance system.
Access the system interface (if applicable) for real-time visualization and alerts.
Detailed usage instructions and configuration steps are provided in the project documentation.

Contributing
We welcome contributions! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features.

License
This project is licensed under the MIT License (see LICENSE file for details).   

Acknowledgments:
YOLOv5: https://github.com/ultralytics/yolov5
OpenCV: https://opencv.org/   

Sources and related content


