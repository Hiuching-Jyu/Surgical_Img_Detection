import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# Step 1: Load model
sam_checkpoint = "/home/hiuching-g/PRHK/segment-anything/weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Step 2: Load image
image_path = "/home/hiuching-g/PRHK/test_images/surgery07_2615.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Create predictor
predictor = SamPredictor(sam)
predictor.set_image(image)

# Step 4: Provide a prompt (e.g., a point)
input_point = np.array([[500, 400]])  # the point you want to segment
input_label = np.array([1])           # 1 represents foreground，0 represents background

# Step 5: Predict mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# Step 6: Visualize and save the first mask
mask = masks[0]
masked_image = image.copy()
masked_image[~mask] = 0

cv2.imwrite("/home/hiuching-g/PRHK/Output_seg_anything/seg_surgery07_2615.jpg", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
print("✅ Mask saved to seg_surgery07_2615.jpg")

