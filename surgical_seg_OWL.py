import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append("/home/hiuching-g/PRHK/segment-anything")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# === Paths ===
input_dir = "/home/hiuching-g/PRHK/test_images"
output_seg_folder = "/home/hiuching-g/PRHK/Output_seg_anything"
output_joint_folder = "/home/hiuching-g/PRHK/Output_joint_seg"
os.makedirs(output_seg_folder, exist_ok=True)
os.makedirs(output_joint_folder, exist_ok=True)

# === OWL-ViT ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# === SAM Model ===
sam_checkpoint = "/home/hiuching-g/PRHK/segment-anything/weights/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,           # 默认是 32，调低减少计算
    pred_iou_thresh=0.95,         # 提高置信度阈值，减少 mask
    stability_score_thresh=0.95,  # 同上
    crop_n_layers=0,              # 默认 1，设为 0 可关闭分块处理，节省显存
)

# === Prompts ===
prompts = [
    "robotic grasper", "robotic needle holder", "surgical scissors",
    "uterus", "fallopian tube", "ligament",
    "blood vessel", "bleeding site", "surgical tool", "Da Vinci end effector"
]
font = ImageFont.load_default()

# === Loop over images ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, filename)
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)
    max_dim = 512
    h, w = image_np.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # === Step 1: Get SAM masks (no label) ===
    masks = mask_generator.generate(image_np)

    seg_vis = image_np.copy()
    for idx, m in enumerate(masks):
        mask = m["segmentation"]
        # Save mask result
        masked = image_np.copy()
        masked[~mask] = 0
        save_path = os.path.join(output_seg_folder, f"{os.path.splitext(filename)[0]}_mask{idx}.png")
        cv2.imwrite(save_path, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

        # Draw colored on seg_vis for visualization
        seg_vis[mask] = (0, 255, 0)

    # Save overlay of all masks
    overlay_all = os.path.join(output_seg_folder, f"overlay_{filename}")
    cv2.imwrite(overlay_all, cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved all SAM masks: {overlay_all}")

    # === Step 2: Run OWL-ViT on each mask to classify it ===
    for idx, m in enumerate(masks):
        mask = m["segmentation"]
        masked_image = image_np.copy()
        masked_image[~mask] = 0
        masked_pil = Image.fromarray(masked_image)

        inputs = owl_processor(text=prompts, images=masked_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = owl_model(**inputs)
        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=[masked_pil.size[::-1]],
            threshold=0.1        )[0]

        if len(results["scores"]) == 0:
            print(f"❌ No label detected for mask {idx} in image {filename}")
            continue

        # Take top-1 prediction
        top_idx = torch.argmax(results["scores"])
        label = prompts[results["labels"][top_idx]]
        box = [int(x) for x in results["boxes"][top_idx].tolist()]
        score = results["scores"][top_idx].item()

        # Draw and save labeled mask
        output_img = masked_image.copy()
        cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(output_img, f"{label} ({score:.2f})", (box[0], max(0, box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        save_path = os.path.join(
            output_joint_folder,
            f"{os.path.splitext(filename)[0]}_mask{idx}_{label.replace(' ', '_')}.png"
        )
        cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        print(f"🟢 Labeled: {save_path}")
