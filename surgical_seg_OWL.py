import os
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === 路径设置 ===
input_dir = "/home/hiuching-g/PRHK/test_images"
output_seg_folder = "/home/hiuching-g/PRHK/Output_seg_anything"
output_owl_only_folder = "/home/hiuching-g/PRHK/Output_OWL_ViT0.2"
output_joint_folder = "/home/hiuching-g/PRHK/Output_joint_seg_overlay"

os.makedirs(output_seg_folder, exist_ok=True)
os.makedirs(output_joint_folder, exist_ok=True)

# === 初始化 SAM ===
device = "cuda"
sam_checkpoint = "/home/hiuching-g/PRHK/segment-anything/weights/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=8,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.95,
    crop_n_layers=0,
)

# === 主循环：处理每张图像 ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    input_path = os.path.join(input_dir, filename)
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)

    # === 获取 SAM mask 并绘制绿色框 ===
    masks = mask_generator.generate(image_np)
    for idx, m in enumerate(masks):
        mask = m["segmentation"]
        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            continue
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        y_min, y_max = np.min(y_idx), np.max(y_idx)

        # === 加载 OWL-ViT 可视化结果图 ===
        owl_path = os.path.join(output_owl_only_folder, f"OWL_{filename}")
        if not os.path.exists(owl_path):
            print(f"[!] OWL result not found: {owl_path}")
            continue
        owl_img = cv2.imread(owl_path)
        if owl_img is None:
            print(f"[!] Failed to read: {owl_path}")
            continue

        # === 叠加绿色框（SAM mask） ===
        cv2.rectangle(owl_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # === 保存对比图 ===
        save_path = os.path.join(output_joint_folder, f"{os.path.splitext(filename)[0]}_mask{idx}_sam_in_owl.png")
        cv2.imwrite(save_path, owl_img)
        print(f"[✔] Overlay saved: {save_path}")
