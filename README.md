# Yolo Web app using Streamlit and OpenCV

Object detection models, powered by neural networks, play a crucial role in computer vision by accurately identifying and categorizing objects within images. 
In this project, we leverage YOLOv8 (You Only Look Once version 8), a highly regarded model known for its speed and precision in real-time object detection tasks. 
Our goal is to develop a user-friendly web application using Streamlit, a Python framework for creating interactive web apps, combined with OpenCV, a versatile computer vision library.

The application we're building allows users to upload videos, enabling the YOLOv8 model to analyze each frame and detect objects of interest. 
Detected objects are visually marked with bounding boxes and labeled with information such as object class and detection confidence. 
Key features of our application include a file uploader for seamless video input, the ability to select specific object classes for detection, and real-time feedback on 
detection metrics. Users have the flexibility to choose which object classes they want the model to detect, tailoring the application's functionality to their specific needs.

By integrating these tools, our project not only showcases the capabilities of advanced object detection technologies but also demonstrates practical applications in fields 
such as surveillance, automated systems, and video analytics. This application serves as a robust platform for exploring YOLOv8's capabilities in identifying and tracking 
objects across various video scenarios, enhancing accessibility and usability through a straightforward and intuitive interface.

## Image Detection Section:
The image detection section of this project allows users to upload images for object detection using the YOLOv8 model. Upon uploading an image, users can initiate detection by 
clicking the "Process Image" button. The YOLOv8 model processes the image and identifies objects based on predefined classes. Detected objects are highlighted with bounding boxes 
and labeled with their respective class names and confidence scores. This section is designed to provide accurate and efficient object detection results from static images, making 
it suitable for applications such as analyzing photographs for specific objects or scenarios.

## Video Detection Section:
In the video detection section, users can upload video files in MP4 format to perform object detection using YOLOv8. After uploading, the application processes each frame of the video, 
detecting and annotating objects in real-time. Detected objects are outlined with bounding boxes and labeled similarly to the image detection section. Users can visualize the processed 
video with overlaid annotations, providing a comprehensive view of object movements and interactions over time. This functionality is ideal for tasks such as surveillance monitoring, 
where continuous analysis of video feeds is necessary to ensure real-time object detection and tracking.

## Webcam Detection Section:
The webcam detection section offers real-time object detection directly from a live webcam feed. Users can initiate detection by clicking the "Start Webcam Detection" button, 
which activates the webcam to capture live video. Each frame from the webcam stream undergoes object detection using YOLOv8, with detected objects displayed in real-time on the 
screen. Similar to the video and image sections, detected objects are highlighted with bounding boxes and labeled to indicate their class and confidence score. This section provides 
immediate feedback and is suitable for applications requiring instant analysis of live video streams, such as security monitoring or interactive systems that respond to real-world 
events in real-time.
