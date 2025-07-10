import os
import cv2
import numpy as np
import torch
import json
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# === 路径设置 ===
input_dir = "/home/hiuching-g/PRHK/test_images"
output_owl_only_folder = "/home/hiuching-g/PRHK/Output_OWL_ViT0.2"
output_joint_folder = "/home/hiuching-g/PRHK/Output_joint_seg_owl_qwen"

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

# === 初始化 QWEN ===
qwen_path = "/home/hiuching-g/PRHK/Qwen"
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=torch.float16).to("cpu")
qwen_processor = AutoProcessor.from_pretrained(qwen_path)

# === IoU 函数 ===
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# === 主循环 ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    input_path = os.path.join(input_dir, filename)
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)

    # === 获取 SAM masks ===
    masks = mask_generator.generate(image_np)
    for idx, m in enumerate(masks):
        mask = m["segmentation"]
        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            continue
        sam_box = [np.min(x_idx), np.min(y_idx), np.max(x_idx), np.max(y_idx)]

        # === 加载 OWL 框图像（推断红框位置）===
        owl_path = os.path.join(output_owl_only_folder, f"OWL_{filename}")
        if not os.path.exists(owl_path):
            print(f"[!] OWL result not found: {owl_path}")
            continue
        owl_img = cv2.imread(owl_path)
        if owl_img is None:
            print(f"[!] Failed to read: {owl_path}")
            continue

        red_mask = cv2.inRange(owl_img, (0, 0, 200), (100, 100, 255))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        matched = False
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            owl_box = [x, y, x + w, y + h]
            iou = calculate_iou(sam_box, owl_box)
            if iou > 0.3:
                matched = True
                break

        if matched:
            vis_img = owl_img.copy()
            cv2.rectangle(vis_img, (sam_box[0], sam_box[1]), (sam_box[2], sam_box[3]), (0, 255, 0), 2)

            # === QWEN 推理识别步骤 ===
            # qwen_prompt = "Which step is currently shown in this surgical scene? Return short answer only."
            qwen_prompt = """This image is from a robot-assisted hysterectomy performed with the da Vinci system.

            The surgery consists of the following key steps:
            1. Port Placement and Docking  
            2. Exposure and Inspection  
            3. Uterine Mobilization (Round and Broad Ligament Dissection)  
            4. Vessel Control (Uterine Artery and IP Ligament Ligation)  
            5. Bladder Dissection and Bladder Flap Creation  
            6. Colpotomy (Cutting around the Cervix/Vaginal Junction)  
            7. Uterus Removal  
            8. Vaginal Cuff Closure  
            9. Final Hemostasis and Robot Exit
            
            Question:  
            **Which step is currently shown in this surgical image?**  
            Return only the **most relevant step name** (from the list above).  
            Do not explain. Do not number. Return a **short text label only**.
            """
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qwen_prompt}
                                                     ]}]
            text_prompt = qwen_processor.apply_chat_template(messages, add_generation_prompt=True)
            qwen_inputs = qwen_processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
            with torch.no_grad():
                qwen_output = qwen_model.generate(**qwen_inputs, max_new_tokens=64)
                trimmed_output = qwen_output[:, qwen_inputs["input_ids"].shape[1]:]
                step_text = qwen_processor.batch_decode(trimmed_output, skip_special_tokens=True)[0]

            # === 写入步骤文字到图像左上角 ===
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, step_text.strip(), (10, 30), font, 1, (255, 255, 0), 2)

            # === 保存最终结果图 ===
            save_path = os.path.join(output_joint_folder, f"{os.path.splitext(filename)[0]}_joint.png")
            cv2.imwrite(save_path, vis_img)
            print(f"[✅] Saved: {save_path}")
            torch.cuda.empty_cache()


