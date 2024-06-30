import os
import cv2
import io
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque



def display_yolov8_overview():
    st.markdown("""
    ### YOLOv8: An Overview

    #### Introduction
    YOLO (You Only Look Once) is a family of state-of-the-art object detection models known for their speed and accuracy. YOLOv8 represents the latest iteration in this series, building on the strengths of its predecessors while introducing several enhancements. Developed by Ultralytics, YOLOv8 aims to provide a robust and efficient solution for real-time object detection tasks.

    #### Key Features and Improvements
    - **Enhanced Backbone and Neck**: YOLOv8 incorporates a more efficient backbone and neck architecture, which improves feature extraction and fusion. This results in better detection accuracy and speed.
    - **Advanced Anchor-Free Detection**: Unlike previous versions that relied on predefined anchor boxes, YOLOv8 employs an anchor-free detection mechanism. This reduces the complexity and increases the flexibility of the model, allowing it to adapt better to various object sizes and shapes.
    - **Improved Training Techniques**: YOLOv8 leverages advanced training techniques such as data augmentation, mosaic augmentation, and self-distillation. These techniques help the model generalize better to unseen data and improve its overall performance.
    - **Optimized for Edge Devices**: YOLOv8 is designed to run efficiently on edge devices with limited computational resources. This makes it suitable for deployment in real-time applications such as surveillance, autonomous driving, and mobile applications.
    - **Modular Design**: The architecture of YOLOv8 is highly modular, allowing for easy customization and adaptation to specific use cases. This modularity also facilitates research and development, enabling users to experiment with different components and configurations.

    #### Performance Metrics
    YOLOv8 achieves state-of-the-art performance across various benchmarks. It balances speed and accuracy, making it ideal for applications that require real-time object detection. The model demonstrates significant improvements in mean Average Precision (mAP) compared to previous YOLO versions, while maintaining or even reducing inference times.

    #### Applications
    - **Surveillance Systems**: YOLOv8 can be used in security cameras to detect and track objects such as people, vehicles, and animals in real-time, enhancing the effectiveness of surveillance systems.
    - **Autonomous Vehicles**: In the automotive industry, YOLOv8 is employed to detect pedestrians, other vehicles, and obstacles, contributing to the development of safer and more reliable self-driving cars.
    - **Retail Analytics**: YOLOv8 can analyze video feeds from retail stores to monitor customer behavior, count foot traffic, and manage inventory more efficiently.
    - **Healthcare**: In medical imaging, YOLOv8 assists in identifying and localizing abnormalities such as tumors in X-rays and MRI scans, aiding in early diagnosis and treatment.
    - **Smart Cities**: YOLOv8 plays a crucial role in smart city initiatives by monitoring traffic flow, detecting incidents, and managing public safety in real-time.

    #### Conclusion
    YOLOv8 represents a significant advancement in the field of object detection, combining high accuracy with real-time performance. Its versatile architecture and ability to run on edge devices make it a valuable tool for a wide range of applications. As the technology continues to evolve, YOLOv8 sets a new benchmark for efficient and effective object detection models.
    """)

def display_detection_metrics(result):
    num_objects = len(result[0].boxes)
    st.subheader("Number of Objects Detected")
    st.write(num_objects)

def app():
    st.set_page_config(page_title='Object Detection Web App', layout='wide')
    st.title('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome! Choose the detection type below.')

    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    # Add a sidebar for selecting detection type
    st.sidebar.title("Detection Settings")
    detection_type = st.sidebar.selectbox("Choose Detection Type", ["Image", "Video", "Webcam"])
    selected_objects = st.sidebar.multiselect('Choose objects to detect', object_names, default=['person'])
    min_confidence = st.sidebar.slider('Confidence score', 0.0, 1.0)

    if detection_type == "Image":
        st.header("Image Detection")
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if st.button('Process Image') and uploaded_image is not None:
            image = Image.open(uploaded_image)
            image = np.array(image)
            result = model(image)
            annotated_image = image.copy()
            for detection in result[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name = model.names[cls]
                label = f'{object_name} {score}'

                if model.names[cls] in selected_objects and score > min_confidence:
                    cv2.rectangle(annotated_image, (x0, y0), (x1, y1), (255, 0, 0), 4)
                    cv2.putText(annotated_image, label, (x0, y0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

            st.image(annotated_image, caption='Processed Image')
            display_detection_metrics(result)
            display_yolov8_overview()

    elif detection_type == "Video":
        st.header("Video Detection")
        uploaded_video = st.file_uploader("Upload Video", type=['mp4'])
        if st.button('Process Video') and uploaded_video is not None:
            input_path = os.path.join(os.getcwd(), uploaded_video.name)
            file_binary = uploaded_video.read()
            with open(input_path, "wb") as temp_file:
                temp_file.write(file_binary)

            video_stream = cv2.VideoCapture(input_path)
            width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(video_stream.get(cv2.CAP_PROP_FPS))
            output_path = os.path.join(os.getcwd(), input_path.split('.')[0] + '_output.mp4')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_data = []  # List to store frame number and object count

            with st.spinner('Processing video...'):
                frame_count = 0
                object_count_text = st.empty()
                while True:
                    ret, frame = video_stream.read()
                    if not ret:
                        break
                    result = model(frame)
                    num_objects = 0

                    for detection in result[0].boxes.data:
                        x0, y0 = (int(detection[0]), int(detection[1]))
                        x1, y1 = (int(detection[2]), int(detection[3]))
                        score = round(float(detection[4]), 2)
                        cls = int(detection[5])
                        object_name = model.names[cls]
                        label = f'{object_name} {score}'

                        if model.names[cls] in selected_objects and score > min_confidence:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 4)
                            cv2.putText(frame, label, (x0, y0 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                            num_objects += 1

                    detections = result[0].verbose()
                    cv2.putText(frame, detections, (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # Convert to RGB before writing
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out_video.write(frame)

                    # Update the number of objects detected for the current frame
                    object_count_text.text(f'Objects detected: {num_objects} on frame {frame_count}')

                    # Store frame number and object count
                    frame_data.append([frame_count, num_objects])

                    frame_count += 1

                video_stream.release()
                out_video.release()

            if os.path.exists(output_path):
                st.video(output_path)

            # Create and display the DataFrame
            df = pd.DataFrame(frame_data, columns=['Frame Number', 'Number of Objects'])
            st.write(df)

            # Plot the graph
            fig, ax = plt.subplots()
            ax.plot(df['Frame Number'], df['Number of Objects'], marker='o')
            ax.set_title('Objects Detected per Frame')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Number of Objects')
            st.pyplot(fig)

            display_yolov8_overview()


    elif detection_type == "Webcam":
        st.header("Webcam Detection")
        start_detection = st.button('Start Webcam Detection')

        if start_detection:
            video_stream = cv2.VideoCapture(0)

            stframe = st.empty()

            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break

                result = model(frame)
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'

                    if model.names[cls] in selected_objects and score > min_confidence:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 4)
                        cv2.putText(frame, label, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                frame_with_detections = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_with_detections, channels="RGB")

                num_objects = len(result[0].boxes)
                st.subheader("Number of Objects Detected")
                st.write(num_objects)

            video_stream.release()


if __name__ == "__main__":
    app()
