import os
from ultralytics import YOLO
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
            tracker="bytetrack.yaml"
        )
        print(f"Processed video: {filename}")

print("Tracking completed for all videos.")

# Track with a custom yolov8obb model on a video
# results = model.track(source="test_videos/Speech Therapy Training Session- Moderate to Severe Autism.mp4", show=True, conf=0.3, save=True, tracker="bytetrack.yaml")
