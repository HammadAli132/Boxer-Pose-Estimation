import cv2
import os

VIDEOS_DIR = "data/raw/videos"
FRAMES_DIR = "data/raw/frames"

def list_videos():
    """List available videos in /data/raw/videos"""
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not videos:
        print("❌ No videos found in data/raw/videos/")
        return []
    print("\n🎥 Available videos:")
    for i, v in enumerate(videos, 1):
        print(f"{i}. {v}")
    return videos

def extract_frames(video_path: str, output_dir: str, target_fps: int):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n📦 Total Frames: {total_frames}")
    print(f"⏱️ Default FPS: {original_fps}")

    frame_interval = max(1, original_fps // target_fps)
    frame_count, saved_count = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            saved_count += 1
            frame_filename = f"{video_name}_frame_{saved_count:06d}.png"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"\n✅ Extracted {saved_count} frames at {target_fps} FPS")
    print(f"📂 Saved in: {save_dir}")

def main():
    videos = list_videos()
    if not videos:
        return

    # Ask user for video selection
    choice = input("\nEnter video name or number: ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if idx < 0 or idx >= len(videos):
            print("❌ Invalid number")
            return
        video_file = videos[idx]
    else:
        if choice not in videos:
            print("❌ Invalid video name")
            return
        video_file = choice

    video_path = os.path.join(VIDEOS_DIR, video_file)

    # Show default FPS
    cap = cv2.VideoCapture(video_path)
    default_fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    print(f"Default FPS of {video_file}: {default_fps}")

    try:
        target_fps = int(input("Enter target FPS for frame extraction: ").strip())
    except ValueError:
        print("❌ Invalid FPS input!")
        return

    extract_frames(video_path, FRAMES_DIR, target_fps)

if __name__ == "__main__":
    main()
