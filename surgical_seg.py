import os
import torch
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

# ====== Configuration ======
sam_checkpoint = "/home/hiuching-g/PRHK/segment-anything/weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
input_folder = "/home/hiuching-g/PRHK/test_images/"
output_folder = "/home/hiuching-g/PRHK/Output_seg_anything/"

os.makedirs(output_folder, exist_ok=True)

# ====== Load SAM Model ======
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# ====== Loop Over Images ======
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in image_files:
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to read {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # You can later modify this to use a smarter prompt strategy (like center point or automatic)
    height, width, _ = image.shape
    input_point = np.array([[width // 2, height // 2]])
    input_label = np.array([1])  # foreground

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    mask = masks[0]
    masked_image = image_rgb.copy()
    masked_image[~mask] = 0

    output_path = os.path.join(output_folder, f"seg_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {output_path}")
