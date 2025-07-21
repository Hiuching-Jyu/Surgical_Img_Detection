import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch 
import difflib
# Model initialization, only execute once
MODEL_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)


# Function

def clamp_boxes(boxes, H, W):
    clamped = []
    for x1, x2, y1, y2 in boxes:
        x1c = max(0, min(x1, W-1))
        y1c = max(0, min(y1, H-1))
        x2c = max(0, min(x2, W-1))
        y2c = max(0, min(y2, H-1))
        if x2c < x1c: x1c, x2c = x2c, x1c
        if y2c < y1c: y1c, y2c = y2c, y1c
        clamped.append([x1c, y1c, x2c, y2c])
    if len(clamped) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(clamped)
        
def recognize_step(phrases):
    detected = set()
    for ph in phrases:
        low = ph.lower()
        for kw in STEP_KEYWORDS.keys():
            if difflib.get_close_matches(kw.lower(), [low], cutoff=0.6):
                detected.add(STEP_KEYWORDS[kw])
    return ", ".join(sorted(detected)) if detected else "Unknown"

# Parameter setup
INPUT_DIR     = "/home/hiuching-g/PRHK/test_images"
OUTPUT_DIR = "/home/hiuching-g/PRHK/Output/Output_GroundingDINO_Newprompts"
TEXT_PROMPT = (
"uterus, ovaries, fallopian tubes, bladder, ureter, robotic grasper, scissors, forceps, suction, needle driver"
)
STEP_KEYWORDS = {
    "Port Placement and Docking": "Port Placement and Docking",
    "Exposure and Inspection": "Exposure and Inspection",
    "Uterine Mobilization": "Uterine Mobilization",
    "Vessel Control": "Vessel Control",
    "Bladder Dissection and Bladder Flap Creation": "Bladder Dissection and Bladder Flap Creation",
    "Colpotomy": "Colpotomy",
    "Uterus Removal": "Uterus Removal",
    "Vaginal Cuff Closure": "Vaginal Cuff Closure",
    "Final Hemostasis and Robot Exit": "Final Hemostasis and Robot Exit"
}

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


for fname in os.listdir(INPUT_DIR):
    name, ext = os.path.splitext(fname)
    if ext.lower() not in EXTENSIONS:
        continue

    in_path  = os.path.join(INPUT_DIR,  fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    # 1) Read image and infer
    image_source, image = load_image(in_path)
    H, W = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # 2) cut boxes to fit the image size

    boxes = clamp_boxes(boxes, H, W)
    boxes = torch.from_numpy(boxes).float() 

    # 3) recognize the step from phrases
    step_text = recognize_step(phrases)

    # 4) add step text to the image
    canvas = image_source.copy()
    cv2.rectangle(canvas, (0,0), (W, 30), (0,0,0), -1)
    cv2.putText(canvas, f"Step: {step_text}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # 5) annotate the image with boxes and phrases
    annotated = annotate(image_source=canvas,
                         boxes=boxes,
                         logits=logits,
                         phrases=phrases)

    # 6) save
    cv2.imwrite(out_path, annotated)
    print(f"[OK] {fname} → {out_path} (Step: {step_text})")
    
    
print("All images processed.")

