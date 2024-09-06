# Therapist and Child Detection and Tracking

## Assignment Description

This project aims to develop a detection and tracking pipeline specifically for identifying children and therapists in videos. The goal is to assign unique IDs to each person, track them throughout the video, and handle re-entries, occlusions, and new entries.

The ultimate purpose is to help analyze behavior and engagement levels of children with Autism Spectrum Disorder (ASD) and therapists to create better treatment plans.

## Problem Statement

The challenge is to create an optimized inference pipeline that processes long-duration videos, detects children and adults, assigns unique IDs, and tracks them throughout the video. The pipeline should also be capable of:

- Assigning unique IDs to persons
- Handling re-entries and post-occlusion tracking
- Assigning new IDs to first-time entries in the frame

## Dataset used

The [**child_adult_detection Dataset**](https://universe.roboflow.com/mamatha/child_adult_detection) from Roboflow was used for training, validation, and testing. The dataset contains labeled images for detecting children and adults in video frames.

- **Train Set**: 8162 images (89%)
- **Validation Set**: 762 images (8%)
- **Test Set**: 227 images (2%)

### Sample Code to Download the Dataset

You can use the following Python code to download the dataset using the Roboflow API:

```bash
!pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="Your API KEY")
project = rf.workspace("mamatha").project("child_adult_detection")
version = project.version(14)
dataset = version.download("yolov8-obb")
```

## Models Trained

### Model 1: YOLOv8n-obb

- **Training Time**: 1.047 hours
- **Epochs**: 10
- **Batch Size**: 8
- **Image Size**: 1024x1024
- **Layers**: 187
- **Parameters**: 3,077,609
- **GFLOPs**: 8.3

```python
from ultralytics import YOLO
# Load YOLOv8n-obb model
model = YOLO("yolov8n-obb.pt")
# Train the model
results = model.train(data="/content/child_adult_detection-14/data.yaml", epochs=10, imgsz=1024, batch=8)

```

### Model 2: YOLOv8m-obb

- **Training Time**: 1.984 hours
- **Epochs**: 10
- **Batch Size**: 8
- **Image Size**: 1024x1024
- **Layers**: 237
- **Parameters**: 26,401,225
- **GFLOPs**: 80.8

```python
from ultralytics import YOLO
# Load YOLOv8m-obb model
model = YOLO("yolov8m-obb.pt")
# Train the model
results = model.train(data="/content/child_adult_detection-14/data.yaml", epochs=10, imgsz=1024, batch=8)
```

## Performance and Model Evaluation

### Model 1: YOLOv8n-obb

- **mAP50**: 0.913 (overall)
- **mAP50-95**: 0.683 (overall)
- **Adult Class mAP50-95**: 0.707
- **Child Class mAP50-95**: 0.659
- **Inference Speed**: 4.8ms per image

### Model 2: YOLOv8m-obb

- **mAP50**: 0.919 (overall)
- **mAP50-95**: 0.71 (overall)
- **Adult Class mAP50-95**: 0.741
- **Child Class mAP50-95**: 0.679
- **Inference Speed**: 20.1ms per image

## Logic Behind Analyzing Model Predictions

### Step-by-Step Breakdown:

1. **Loading the Trained Model**:
    - The pretrained models are loaded using the YOLO framework. Each model was trained on the dataset and evaluated using metrics such as mAP50 and mAP50-95.
2. **Inference Process**:
    - Inference is performed on the test videos using the `model.track()` function. For each frame, bounding boxes are generated with associated labels ("Child" or "Adult") and confidence scores. The `conf=0.3` parameter ensures that only confident predictions are used.
3. **Assigning Unique IDs**:
    - Each detected individual is assigned a unique ID using the default **ByteTrack** tracker. This helps maintain consistent tracking across frames, ensuring IDs are retained even after temporary occlusions.
4. **Tracking Performance**:
    - The **ByteTrack** tracker used here performs reasonably well for most scenarios. However, for better tracking accuracy and efficiency, especially in challenging conditions like occlusions and re-entries, the **DeepSORT** tracker is recommended. DeepSORT offers superior re-identification and tracking robustness, which can be easily integrated with this model for enhanced performance.
5. **Post-Processing**:
    - Bounding boxes and unique IDs are visualized on the video frames, providing a clear display of the detected individuals' movement patterns. The final output videos are saved with these visual annotations for further review and analysis.

---

## Inference Code for Tracking

The following code was used for inference on the test videos. It uses **ByteTrack** as the tracker, but to improve tracking accuracy, **DeepSORT** can be substituted for better performance in challenging scenarios.

```python
import os
from ultralytics import YOLO

# Load the custom model
model = YOLO("customyolov8nobb.pt")

# Define the folder containing the videos
video_folder = 'test_videos'

# Iterate over all files in the video folder
for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_folder, filename)

        # Track with the YOLO model
        results = model.track(
            source=video_path,
            show=True,
            conf=0.3,
            save=True,
            tracker="bytetrack.yaml"  # Can be replaced with 'deepsort.yaml' for better performance
        )
        print(f"Processed video: {filename}")

print("Tracking completed for all videos.")
```

## Downloading Test Videos

The test videos were downloaded using `yt-dlp` with the following code:

```bash
!pip install yt-dlp
```

```python
import os
from yt_dlp import YoutubeDL

# Create a folder for videos
if not os.path.exists("test_videos"):
    os.makedirs("test_videos")

# Download videos using yt-dlp
def download_videos(video_list_file):
    with open(video_list_file, "r") as file:
        video_urls = file.readlines()

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': 'videos/%(title)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        for video_url in video_urls:
            ydl.download([video_url.strip()])

# Download videos
download_videos('test_videos.txt')

```

The zip file includes:

- `test_videos.txt` for downloading test videos.
- Trained models (`customyolov8nobb.pt`, `customyolov8mobb.pt`).
- The `requirements.txt` file.
- Output videos and the inference script.

---

## Results

Here are sample results from both models:

- **YOLOv8n-obb Results**:

![yolov8nobb results.png](Therapist%20and%20Child%20Detection%20and%20Tracking/yolov8nobb_results.png)

- **YOLOv8m-obb Results**:

![yolov8mobb results.png](Therapist%20and%20Child%20Detection%20and%20Tracking/yolov8mobb_results.png)

---

## Improvements and Future Work

1. **Larger Dataset**:
    - To further improve model accuracy, training on a much larger dataset is recommended. The current model was trained on a limited dataset and using the free-tier T4 GPU in Google Colab, which restricted the number of epochs and performance.
2. **Training with More Epochs**:
    - More epochs can be used to fine-tune the model, especially for harder-to-detect scenarios like occlusions and re-entries.
3. **Improved Tracking with DeepSORT**:
    - Although **ByteTrack** was used in the current setup, integrating the **DeepSORT** tracker can improve tracking accuracy. DeepSORT allows for more accurate re-identification, which is essential for maintaining IDs during post-occlusion tracking and in multi-person scenarios.

---

## Usage Instructions

1. Install the dependencies listed in `requirements.txt`.
2. Download the dataset using the provided sample code.
3. Train the model or use the pretrained weights to run inference.
4. Download the test videos using the `test_videos.txt` file.
5. Run inference on the test videos.
6. Review the output videos with overlaid bounding boxes and unique IDs for detected persons.

---

This README provides comprehensive instructions to reproduce the results, as well as suggestions for future improvements. The results section will display the performance of both YOLO models used.#   T h e r a p i s t - a n d - C h i l d - D e t e c t i o n - a n d - T r a c k i n g  
 #   P e r s o n T r a c k e r  
 