# Real-Time License Plate Anonymizer

This project is a high-performance computer vision pipeline designed to automatically detect, track, and anonymize vehicle license plates in real-time. It serves as a practical tool for ensuring privacy compliance and promoting ethical surveillance in various applications, from public street monitoring to private data collection.

An animated demonstration of the real-time pipeline is available. A key snapshot from this demo, which clearly shows the blurring and anonymization effect on a detected license plate, can be seen in the image file named `license_anonymizer_demo_snapshot.png`.

## Project Overview

In an era of increasing video surveillance, protecting individual privacy is paramount. This project addresses this need by creating a system that can process video feeds (live or recorded) and automatically obscure sensitive license plate information. The system is built to be robust and efficient, leveraging state-of-the-art deep learning models for detection and tracking.

## Key Features

-   **High-Accuracy Detection:** Utilizes a **YOLOv8** model custom-trained on a specific license plate dataset for precise and reliable detection.
-   **Persistent Multi-Object Tracking:** Implements **DeepSort** to assign and maintain a unique ID for each detected plate, ensuring consistent tracking across frames even through temporary occlusions.
-   **Real-Time Anonymization:** Applies a blurring filter using **OpenCV** to the tracked regions, effectively anonymizing the plates.
-   **Versatile Input:** Supports processing from multiple sources, including static images, video files, and live webcam feeds.
-   **Clean Output:** The final output is a clean video with blurred plates, without any distracting bounding boxes, ready for professional use.

## Tech Stack

-   **Backend:** Python
-   **Deep Learning:** PyTorch, Ultralytics (for YOLOv8)
-   **Computer Vision:** OpenCV
-   **Object Tracking:** DeepSort

---

## The Core Pipeline Explained

The application functions as a multi-stage pipeline, where the output of each stage serves as the input for the next.

### 1. Custom Model Training (YOLOv8)

To achieve high accuracy, a standard pre-trained object detector is insufficient. A **YOLOv8** model was fine-tuned on a custom-annotated dataset of vehicle license plates. This process involved:
-   Gathering and labeling a diverse dataset of images containing license plates.
-   Training the YOLOv8 architecture to specifically recognize the features and patterns of these plates.
-   Exporting the final trained model weights (`best.pt`) for use in the inference pipeline.

### 2. Frame-by-Frame Detection

The inference process begins by feeding a video stream (or a single image) into the pipeline. For each frame, the custom-trained YOLOv8 model is used to perform object detection, identifying the locations of all visible license plates and outputting their bounding box coordinates `(x, y, w, h)`.

### 3. Multi-Object Tracking (DeepSort)

While detection identifies plates in a single frame, tracking is required to maintain the identity of each plate over time. This is where **DeepSort** comes in.
-   The bounding boxes from YOLO are passed to the DeepSort tracker.
-   DeepSort uses a combination of a Kalman filter for motion prediction and a deep association metric to assign a unique and persistent ID to each license plate.
-   This ensures that a specific car's license plate is treated as the same object from frame to frame, even if the YOLO detector momentarily fails to see it.

### 4. Anonymization (Blurring)

With a consistent, tracked bounding box for each license plate, the final step is anonymization.
-   For each frame, the pipeline iterates through the currently tracked objects.
-   It extracts the Region of Interest (ROI) defined by the object's bounding box.
-   A **Gaussian Blur** filter from OpenCV is applied to this ROI.
-   The blurred ROI is then placed back onto the original frame.
-   The final frame, with all tracked license plates blurred and no bounding boxes drawn, is then saved or displayed.

---
The source code for this project is available on my GitHub profile (Shiv-Expert2503) in the 'License_Blur' repository.

