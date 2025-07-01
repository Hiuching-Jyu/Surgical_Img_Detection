import os
import cv2
import torch
from PIL import Image
from maskrcnn_benchmark.config import cfg
#from glip_demo import GLIPDemo  # assuming GLIPDemo is imported from your GLIP repo
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


# 1. Load model configuration and weights
CFG_PATH    = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
WEIGHT_PATH = "MODEL/glip_tiny_model_o365_goldg.pth"

cfg.merge_from_file(CFG_PATH)
cfg.merge_from_list(["MODEL.WEIGHT", WEIGHT_PATH])
cfg.merge_from_list(["MODEL.DEVICE", "cuda" if torch.cuda.is_available() else "cpu"])
cfg.freeze()

glip = GLIPDemo(
    cfg,
    min_image_size=320,
    confidence_threshold=0.35
)

# 2. Directory and prompt setup
INPUT_DIR   = "/home/hiuching-g/PRHK/test_images"
OUTPUT_DIR  = "/home/hiuching-g/PRHK/Output_Glip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_PROMPT = (
    "Robotic grasper . monopolar curved scissors . vessel sealer . suction . "
    "Hook cautery . fenestrated forceps . Bleeding site . necrotic tissue . "
    "enlarged uterus . adhesion . fatty tissue . cystic lesion . vessel proximity . "
    "tissue rupture "
)

# 3. Inference and annotation
for file_name in os.listdir(INPUT_DIR):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(INPUT_DIR, file_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #predictions = glip.run_on_web_image(image_rgb, TEXT_PROMPT)
    result_img, boxlist = glip.run_on_web_image(image_rgb, TEXT_PROMPT)
    
    #print(type(predictions))
    #print(predictions)
    #boxes = predictions["boxes"]
    #labels = predictions["labels"]
    boxes  = boxlist.bbox.cpu().numpy()   # shape: (N, 4)
    labels = boxlist.get_field("labels") 
    
    #if hasattr(glip, "entities") and glip.entities:  # e.g., glip.entities is the prompt list
    #    print("labels:", labels)
    #    print("entities:", glip.entities)
    #    print("len(entities):", len(glip.entities))
    #    labels = [glip.entities[i] for i in labels]
    
    raw_labels = boxlist.get_field("labels").tolist()

    if hasattr(glip, "entities") and glip.entities:
        entities = glip.entities
        mapped_labels = [
            entities[i] if 0 <= i < len(entities) else f"unknown_{i}"
            for i in raw_labels
        ]
    else:
        mapped_labels = [str(i) for i in raw_labels]  # fallback: just use index
    # Draw each box and label
    #for box, label in zip(boxes, labels):
     #   x1, y1, x2, y2 = map(int, box)
     #   cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
     #   cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
     #               fontScale=0.6, color=(0, 0, 255), thickness=2)
    
    for box, label in zip(boxes, mapped_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.putText(image, label, (x1, max(0, y1 - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(0, 0, 255), thickness=2)
                
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file_name)[0] + ".png")
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")

