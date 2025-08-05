import dashscope
from dashscope import MultiModalConversation
import os
import re
import json
from PIL import Image, ImageDraw, ImageFont
import base64
# from rag_module import retrieve_step_by_rag   # ← removed: no RAG

# <editor-fold desc=" 1. Initialization for API-Key and dirs">
dashscope.api_key = 'sk-8694ac696f1c42aba2f1cfb254a5918d'  # Replace with your key

image_dir = "/home/hiuching-g/PRHK/test_images_236"
output_dir = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_withoutRAG_236"
os.makedirs(output_dir, exist_ok=True)

# === COCO category mapping (IDs are stable and unique) ===
CATEGORY_MAP = {
    # instruments
    "Bipolar Forceps": 1,
    "Monopolar Scissors": 2,
    "Suction Irrigator": 3,
    "Needle Driver": 4,
    "Cautery Hook": 5,
    "UnclearInstrument": 6,
    # tissues
    "Uterus": 101,
    "Ovaries": 102,
    "Fallopian Tubes": 103,
    "Bladder": 104,
    "Ureter": 105,
    "UnclearBodyTissue": 106,
}

INSTRUMENT_LABELS = {
    "Bipolar Forceps", "Monopolar Scissors", "Suction Irrigator",
    "Needle Driver", "Cautery Hook", "UnclearInstrument"
}
TISSUE_LABELS = {
    "Uterus", "Ovaries", "Fallopian Tubes", "Bladder", "Ureter", "UnclearBodyTissue"
}

STEP_LABELS = [
    "Preparation & Exposure",
    "Dissection & Vessel Control",
    "Uterus Removal & Closure"
]

input_text = (
    "You are given a robotic surgical image. Perform the following:\n\n"
    "1. Detect and localize **surgical instruments**. For each bounding box, assign exactly one of the following specific labels:\n"
    "- Bipolar Forceps\n"
    "- Monopolar Scissors\n"
    "- Suction Irrigator\n"
    "- Needle Driver\n"
    "- Cautery Hook\n"
    "- UnclearInstrument (if not identifiable)\n\n"
    "2. Detect and localize **body tissues**. For each bounding box, choose only the most likely label for each tissue object from the following options:\n"
    "- Uterus\n"
    "- Ovaries\n"
    "- Fallopian Tubes\n"
    "- Bladder\n"
    "- Ureter\n"
    "- UnclearBodyTissue (if not identifiable)\n\n"
    "Make sure you respond in **JSON only**, and ensure the format is correct:\n"
    "{\n"
    "  \"bboxes\": [\n"
    "    {\"label\": \"<specific label>\", \"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int},\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Try to identify at least 3 items for an image.\n"
)

# === helpers ===
def xyxy_to_xywh(x1, y1, x2, y2):
    x = min(x1, x2); y = min(y1, y2)
    w = max(0, abs(x2 - x1)); h = max(0, abs(y2 - y1))
    return [int(x), int(y), int(w), int(h)]

def label_to_category_id(label: str):
    return CATEGORY_MAP.get(label, None)

def safe_float(v, default=1.0):
    try:
        return float(v)
    except Exception:
        return default

def extract_json_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\{[\s\S]*\}', text)
    return m.group(0) if m else ""

def try_parse_json(s: str):
    if not s:
        return None
    t = s.strip()
    # tolerant cleanup
    t = re.sub(r',(\s*[}\]])', r'\1', t)                         # trailing comma
    t = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*):', r'\1"\2"\3:', t) # unquoted keys
    t = re.sub(r"(?<!\\)'", '"', t)                              # single -> double quotes
    try:
        return json.loads(t)
    except Exception:
        return None

# collectors
coco_detections = []
step_predictions = []

# </editor-fold>

for image_file in os.listdir(image_dir):
    try:
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, image_file)
        # image_id = image_file
        image_id = os.path.splitext(image_file)[0].split("_")[0]   # keep only the 'xxx' part
        print(f"\n🔍 Processing {image_file}...")

        # Read image (skip if failed)
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Cannot read image {image_path}: {e}. Skipping.")
            step_predictions.append({
                "image_id": image_id,
                "step_top1": "Unknown",
                "step_probs": {
                    "Preparation & Exposure": 1/3,
                    "Dissection & Vessel Control": 1/3,
                    "Uterus Removal & Closure": 1/3
                },
                "note": "image_read_fail"
            })
            continue

        # === 1) Qwen for bboxes (unchanged) ===
        bboxes = []
        try:
            messages_bbox = [{"role": "user", "content": [{"image": image_path}, {"text": input_text}]}]
            response_bbox = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages_bbox,
                temperature=0.4,
                result_format="message",
                vl_high_resolution_images=True,
                timeout=90
            )
            response_text = response_bbox.output.choices[0].message.content if response_bbox and response_bbox.output else ""
            if isinstance(response_text, list):
                response_text = response_text[0].get("text", "")
            elif isinstance(response_text, dict):
                response_text = response_text.get("text", "")
            json_str = extract_json_block(response_text)
            parsed = try_parse_json(json_str)
            if parsed and isinstance(parsed, dict):
                bboxes = parsed.get("bboxes", [])
            else:
                print("⚠️ JSON parsing failed (bbox).")
        except Exception as e:
            print(f"⚠️ BBOX prediction error: {e}. Continue without boxes.")

        # === 2) Qwen for STEP (RAG removed; Qwen outputs step JSON directly) ===
        step_name = "Unknown"
        step_probs = {k: 0.0 for k in STEP_LABELS}
        try:
            # Ask Qwen to return step in strict JSON
            step_prompt = (
                "You are an expert in hysterectomy. "
                "Given the image, classify the current surgical step. "
                "Return JSON ONLY with this exact schema:\n"
                "{\n"
                "  \"step_top1\": \"<one of: Preparation & Exposure | Dissection & Vessel Control | Uterus Removal & Closure>\",\n"
                "  \"step_probs\": {\n"
                "    \"Preparation & Exposure\": float,\n"
                "    \"Dissection & Vessel Control\": float,\n"
                "    \"Uterus Removal & Closure\": float\n"
                "  }\n"
                "}\n"
                "Probabilities should sum to 1. No extra text."
            )
            messages_step = [{"role": "user", "content": [{"image": image_path}, {"text": step_prompt}]}]
            response_step = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages_step,
                temperature=0.2,                 # lower temperature for stable JSON/probs
                result_format="message",
                vl_high_resolution_images=True,
                timeout=90
            )
            step_text = response_step.output.choices[0].message.content if response_step and response_step.output else ""
            if isinstance(step_text, list):
                step_text = step_text[0].get("text", "")
            elif isinstance(step_text, dict):
                step_text = step_text.get("text", "")

            step_json = try_parse_json(extract_json_block(step_text))
            if isinstance(step_json, dict):
                # read top1 and probs if present
                cand = step_json.get("step_top1")
                probs = step_json.get("step_probs", {})
                if isinstance(cand, str) and cand in STEP_LABELS:
                    step_name = cand
                # validate probs
                if isinstance(probs, dict):
                    for k in STEP_LABELS:
                        v = probs.get(k, 0.0)
                        try:
                            step_probs[k] = float(v)
                        except Exception:
                            step_probs[k] = 0.0
                # if probs invalid, fallback to one-hot
                if sum(step_probs.values()) <= 0:
                    step_probs = {k: 0.0 for k in STEP_LABELS}
                    if step_name in step_probs:
                        step_probs[step_name] = 1.0
            else:
                # fallback: one-hot Unknown
                step_name = "Unknown"
                step_probs = {
                    "Preparation & Exposure": 1/3,
                    "Dissection & Vessel Control": 1/3,
                    "Uterus Removal & Closure": 1/3
                }
        except Exception as e:
            print(f"⚠️ Step prediction error: {e}. Use fallback.")
            step_name = "Unknown"
            step_probs = {
                "Preparation & Exposure": 1/3,
                "Dissection & Vessel Control": 1/3,
                "Uterus Removal & Closure": 1/3
            }

        # record step prediction (same structure as before)
        step_predictions.append({
            "image_id": image_id,
            "step_top1": step_name,
            "step_probs": step_probs
        })

        # === 3) Visualization & COCO detections (unchanged) ===
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
            font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)

            def parse_coord(val):
                if isinstance(val, int): return val
                if isinstance(val, float): return int(val)
                if isinstance(val, list) and val:
                    try: return int(float(val[0]))
                    except Exception: return 0
                if isinstance(val, str):
                    nums = re.findall(r'-?\d+\.?\d*', val)
                    try: return int(float(nums[0])) if nums else 0
                    except Exception: return 0
                return 0

            for box in bboxes:
                try:
                    x1 = parse_coord(box.get("x1"))
                    y1 = parse_coord(box.get("y1"))
                    x2 = parse_coord(box.get("x2"))
                    y2 = parse_coord(box.get("y2"))
                    label = box.get("label", "Unknown")

                    xywh = xyxy_to_xywh(x1, y1, x2, y2)
                    cat_id = label_to_category_id(label)
                    score = safe_float(box.get("score", 1.0))

                    if cat_id is None:
                        print(f"⚠️ Unknown label '{label}' — skipping COCO record.")
                    else:
                        coco_detections.append({
                            "image_id": image_id,
                            "category_id": cat_id,
                            "category_name": label,
                            "bbox": xywh,
                            "score": score
                        })

                    color = "green" if any(
                        k in label.lower() for k in ["uterus", "ovaries", "tubes", "bladder", "ureter"]) else "red"
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, max(y1 - 12, 0)), label, fill=color, font=font_def)
                except Exception as e:
                    print(f"⚠️ Failed to draw/record box: {box}, Error: {e}")

            draw.text((50, 20), f"Step: {step_name}", fill="blue", font=font_lar)
            output_path = os.path.join(output_dir, f"annotated_{image_file}")
            try:
                img.save(output_path)
                print(f"✅ Annotated image saved to: {output_path}")
            except Exception as e:
                print(f"⚠️ Failed to save annotated image for {image_file}: {e}")
        except Exception as e:
            print(f"⚠️ Visualization error on {image_file}: {e}. Continue.")

    except Exception as e:
        print(f"❌ Unexpected error for {image_file}: {e}. Skipping this image.")
        step_predictions.append({
            "image_id": image_file,
            "step_top1": "Unknown",
            "step_probs": {
                "Preparation & Exposure": 1/3,
                "Dissection & Vessel Control": 1/3,
                "Uterus Removal & Closure": 1/3
            },
            "note": "outer_try_except"
        })
        continue

# === After processing all images: write outputs ===
detections_path = os.path.join(output_dir, "detections_coco.json")
steps_path = os.path.join(output_dir, "steps_predictions.json")

try:
    with open(detections_path, "w", encoding="utf-8") as f:
        json.dump(coco_detections, f, ensure_ascii=False, indent=2)
    print(f"💾 COCO detections written to: {detections_path}")
except Exception as e:
    print(f"⚠️ Failed to write detections JSON: {e}")

try:
    with open(steps_path, "w", encoding="utf-8") as f:
        json.dump(step_predictions, f, ensure_ascii=False, indent=2)
    print(f"💾 Step predictions written to: {steps_path}")
except Exception as e:
    print(f"⚠️ Failed to write steps JSON: {e}")
