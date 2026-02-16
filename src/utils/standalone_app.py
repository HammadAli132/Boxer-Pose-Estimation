import os
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# ---------------------------------
# Your directory structure
# ---------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"
# ---------------------------------------------------------
#  YOUR VISUALIZATION SETTINGS (unchanged)
# ---------------------------------------------------------
BOX_THICKNESS = 2
KEYPOINT_RADIUS = 5
SKELETON_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
model_name = "yolov11m-pose"

# Custom 14-keypoint skeleton (Correct Structure)
SKELETON_CONNECTIONS = [
    (0, 1),           # Nose -> Neck
    (1, 2), (1, 3),   # Neck -> Shoulders
    (2, 4), (4, 6),   # Left Arm
    (3, 5), (5, 7),   # Right Arm
    (2, 8), (3, 9),   # Torso (Shoulders -> Hips)
    (8, 9),           # Hip connection
    (8, 10), (10, 12),# Left Leg
    (9, 11), (11, 13) # Right Leg
]

# ---------------------------------
# GUI Application
# ---------------------------------
class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 Pose Extractor")
        self.video_path = None

        tk.Button(root, text="Select Video", command=self.select_video, width=30).pack(pady=10)
        tk.Button(root, text="Run Pose Extraction", command=self.run_inference, width=30).pack(pady=10)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if self.video_path:
            messagebox.showinfo("Selected", f"Video selected:\n{self.video_path}")

    def run_inference(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first.")
            return

        # Ask if user wants to process complete video or frame range
        process_complete = messagebox.askyesno(
            "Processing Mode",
            "Do you want to process the COMPLETE video?\n\n"
            "Click 'Yes' for complete video\n"
            "Click 'No' to specify frame range"
        )

        if process_complete:
            start_frame = 0
            end_frame = None  # None indicates process until end
        else:
            # Ask for frame range
            start_frame = simpledialog.askinteger("Start Frame", "Enter start frame:", minvalue=0)
            end_frame = simpledialog.askinteger("End Frame", "Enter end frame:", minvalue=start_frame+1)

            if start_frame is None or end_frame is None:
                return

        # Ask for output save path
        output_path = filedialog.asksaveasfilename(
            title="Save Output Video As...",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not output_path:
            return

        messagebox.showinfo("Processing", "Running pose estimation. Please wait...")

        try:
            self.process_video(start_frame, end_frame, output_path)
            messagebox.showinfo("Done", "Pose video created successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------------------------------------------------
    #  DRAWING FUNCTIONS (unchanged from your code)
    # ---------------------------------------------------------
    def _draw_keypoints(self, frame: np.ndarray, annotation: dict) -> np.ndarray:
        keypoints = annotation['keypoints']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        # Colors per boxer category
        if category_id == 1:
            skeleton_color = (255, 100, 0)
            bbox_color = (255, 0, 0)
            text_bg_color = (200, 0, 0)
            class_name = "BLUE"

        elif category_id == 0:
            skeleton_color = (0, 100, 255)
            bbox_color = (0, 0, 255)
            text_bg_color = (0, 0, 200)
            class_name = "RED"

        else:
            skeleton_color = (128, 128, 128)
            bbox_color = (128, 128, 128)
            text_bg_color = (100, 100, 100)
            class_name = "UNKNOWN"

        keypoint_color = (0, 255, 0)
        text_color = (255, 255, 255)

        # Draw bbox - convert from center format (xywh) to corner format
        x_center, y_center, w, h = bbox
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, BOX_THICKNESS)

        # Parse keypoints
        kpts = []
        for i in range(0, len(keypoints), 3):
            x_kp = keypoints[i]
            y_kp = keypoints[i + 1]
            vis = keypoints[i + 2]
            if vis > 0:
                kpts.append((int(x_kp), int(y_kp), vis))
            else:
                kpts.append(None)

        # Draw skeleton
        for (i, j) in SKELETON_CONNECTIONS:
            if i < len(kpts) and j < len(kpts):
                if kpts[i] is not None and kpts[j] is not None:
                    pt1 = (kpts[i][0], kpts[i][1])
                    pt2 = (kpts[j][0], kpts[j][1])
                    cv2.line(frame, pt1, pt2, skeleton_color,
                            SKELETON_THICKNESS, cv2.LINE_AA)

        # Draw keypoints
        for kp in kpts:
            if kp is not None:
                cv2.circle(frame, (kp[0], kp[1]), KEYPOINT_RADIUS,
                        keypoint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (kp[0], kp[1]), KEYPOINT_RADIUS,
                        (0, 0, 0), 1, cv2.LINE_AA)

        # Draw label
        # In _draw_keypoints, update the label:
        track_id = annotation.get('track_id', -1)
        if track_id >= 0:
            label = f"ID:{track_id} | BOXER {class_name} | KPs:{annotation['num_keypoints']}"
        else:
            label = f"BOXER {class_name} | KPs:{annotation['num_keypoints']}"

        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Draw label background
        cv2.rectangle(frame,
                    (x1, y1 - th - bl - 8),
                    (x1 + tw + 4, y1 - 2),
                    text_bg_color,
                    -1)

        cv2.putText(frame, label,
                    (x1 + 2, y1 - bl - 5),
                    FONT, FONT_SCALE,
                    text_color, FONT_THICKNESS, cv2.LINE_AA)

        return frame


    # ---------------------------------------------------------
    #  🔥 UPDATED POSE PROCESSING — FULL INTEGRATION
    # ---------------------------------------------------------
    def process_video(self, start_frame, end_frame, output_path):
        model_path = MODELS_ROOT / "yolov11x-pose" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found:\n{model_path}")

        model = YOLO(str(model_path))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < start_frame:
                frame_idx += 1
                continue
            
            # If end_frame is None, process until video ends
            if end_frame is not None and frame_idx > end_frame:
                break

            # 🔥 TRACKING MODE
            results = model.track(frame, imgsz=640, conf=0.65, persist=True, verbose=False)

            annotated = frame.copy()

            kpts = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # 🔥 Get track IDs (will be None if tracking fails)
            track_ids = results[0].boxes.id
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)

            for i in range(len(kpts)):
                k = kpts[i]
                bbox = boxes[i]
                cls = int(classes[i])
                track_id = int(track_ids[i]) if track_ids is not None else -1

                annotation = {
                    "keypoints": k.flatten().tolist(),
                    "bbox": bbox.tolist(),
                    "num_keypoints": len(k),
                    "category_id": cls if cls in [0, 1] else -1,
                    "track_id": track_id  # 🔥 Add track ID
                }

                annotated = self._draw_keypoints(annotated, annotation)

            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()


# ---------------------------------
# Run the GUI
# ---------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()