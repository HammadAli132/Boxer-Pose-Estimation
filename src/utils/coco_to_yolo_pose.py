import os
import json
from tqdm import tqdm
from typing import Tuple


def coco_to_yolo_keypoints(
    coco_json: str,
    images_dir: str,
    out_dir: str,
    num_keypoints: int = 14,
    clear_out: bool = True,
) -> Tuple[int, int]:
    """
    Convert COCO-style keypoint annotations to YOLOv8 Pose TXT format.

    Args:
        coco_json: Path to COCO annotations JSON
        images_dir: Path to directory containing frames for this split (used only for warnings)
        out_dir: Directory to save YOLO label files (one .txt per image)
        num_keypoints: Number of keypoints expected in your dataset
        clear_out: If True, remove existing files in out_dir before writing

    Returns:
        (num_labels_written, num_images_missing) -- useful for logging
    """
    # load
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id -> image dict for safe lookup (no assumption on ordering)
    id_to_image = {img["id"]: img for img in coco.get("images", [])}

    # prepare out dir
    if clear_out and os.path.exists(out_dir):
        # clear labels to avoid duplicates and stale files
        for fname in os.listdir(out_dir):
            if fname.endswith(".txt"):
                try:
                    os.remove(os.path.join(out_dir, fname))
                except Exception:
                    pass
    os.makedirs(out_dir, exist_ok=True)

    labels_written = 0
    images_missing = set()
    anns_processed = 0

    # iterate annotations
    for ann in tqdm(coco.get("annotations", []), desc=f"Converting COCO -> YOLO ({os.path.basename(out_dir)})"):
        anns_processed += 1
        # skip if no keypoints at all
        if ann.get("num_keypoints", 0) == 0:
            continue

        img_id = ann.get("image_id")
        img_info = id_to_image.get(img_id)
        if img_info is None:
            tqdm.write(f"⚠️ Warning: image_id {img_id} not found in images[]. Skipping annotation id {ann.get('id')}")
            continue

        img_name = img_info.get("file_name")
        img_w = img_info.get("width")
        img_h = img_info.get("height")

        # sanity check widths/heights
        if not img_w or not img_h:
            tqdm.write(f"⚠️ Warning: width/height missing for image {img_name} (id={img_id}). Skipping.")
            continue

        # check image existence (warning only)
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            images_missing.add(img_name)
            tqdm.write(f"⚠️ Image file not found: {img_path} (label will still be created)")

        # compute normalized bbox cx,cy,w,h
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        # ensure keypoints length; pad with zeros if shorter
        kps = ann.get("keypoints", [])
        expected_len = 3 * num_keypoints
        if len(kps) < expected_len:
            # pad with zeros (x=0,y=0,v=0) — preserves indexing
            kps = kps + [0.0] * (expected_len - len(kps))
        elif len(kps) > expected_len:
            # trim extra just in case (rare)
            kps = kps[:expected_len]

        # build keypoint strings normalized
        kp_parts = []
        for i in range(num_keypoints):
            kx = float(kps[3 * i])
            ky = float(kps[3 * i + 1])
            v = int(kps[3 * i + 2])  # visibility
            # normalize coords
            nkx = kx / img_w if img_w else 0.0
            nky = ky / img_h if img_h else 0.0
            kp_parts.append(f"{nkx:.6f} {nky:.6f} {v}")

        # class index: convert COCO category_id -> 0-based class
        # (assumes your COCO category ids align to dataset YAML names)
        class_idx = ann.get("category_id", 1) - 1

        # final line: "class cx cy w h kp1x kp1y v1 kp2x kp2y v2 ..."
        line = f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_parts)

        # label file path
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(out_dir, label_name)

        # append (multiple instances per image ok)
        with open(label_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
            labels_written += 1

    # summary
    print(f"\nConversion summary for {os.path.basename(out_dir)}:")
    print(f"  annotations scanned: {anns_processed}")
    print(f"  label lines written : {labels_written}")
    print(f"  missing image files : {len(images_missing)} (sample up to 5): {list(images_missing)[:5]}")
    if labels_written == 0:
        print("⚠️ No labels written. Check annotations or number of keypoints.")
    return labels_written, len(images_missing)
