import dashscope
from dashscope import MultiModalConversation
import os
import re
import json
from PIL import Image, ImageDraw, ImageFont
import base64
from rag_module import retrieve_step_by_rag
import os
import json
from typing import Optional
from openai import OpenAI
import os
import concurrent.futures



# <editor-fold desc=" 1.1 Initialization: API-Key and dirs">
dashscope.api_key = 
os.environ["OPENAI_API_KEY"] = 

image_dir = "/home/hiuching-g/PRHK/test_images_236"
output_dir = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_RAG_236"
os.makedirs(output_dir, exist_ok=True)
detections_path = os.path.join(output_dir, "detections_coco.json")
steps_path = os.path.join(output_dir, "steps_predictions.json")
done_images_file = os.path.join(output_dir, "done_images.txt")

# </editor-fold>
# <editor-fold desc=" 1.2 Initialization: Find done images and prepare COCO categories">
# Read done images from file， if it exists
if os.path.exists(done_images_file):
    with open(done_images_file, "r", encoding="utf-8") as f:
        done_images = set(line.strip() for line in f if line.strip())
else:
    done_images = set()

# Collect all detections and step predictions
coco_detections = []
step_predictions = []
# </editor-fold>

# <editor-fold desc=" 1.3 Initialization: COCO categories and labels">
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
    "- UnclearInstrument\n\n"
    "2. Detect and localize **body tissues**. For each bounding box, choose only the most likely label for each tissue object from the following options:\n"
    "- Uterus\n"
    "- Ovaries\n"
    "- Fallopian Tubes\n"
    "- Bladder\n"
    "- Ureter\n"
    "- UnclearBodyTissue\n\n"
    "Make sure you respond in **JSON only** and **strictly follow the following format, no sign changed**:\n"
    "{\n"
    "  \"bboxes\": [\n"
    "    {\"label\": \"<specific label>\", \"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int},\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Try to identify at least 3 items for an image.\n"
)
# </editor-fold>
# <editor-fold desc=" 2.1 Functions Setup: Helper functions for bbox and step processing">

def xyxy_to_xywh(x1, y1, x2, y2):
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(0, abs(x2 - x1))
    h = max(0, abs(y2 - y1))
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

# </editor-fold>


# <editor-fold desc=" 2.2 Functions Setup: Qwen bbox and OpenAI JSON fixing functions">
# === 2.2.1 Qwen bbox and OpenAI JSON fixing functions ===
def call_qwen_bbox(image_path: str, input_text: str, temperature: float = 0.4, timeout: int = 90) -> str:
    """Only call Qwen to get raw bbox text."""
    messages_bbox = [{"role": "user", "content": [{"image": image_path}, {"text": input_text}]}]
    resp = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages_bbox,
        temperature=temperature,
        result_format="message",
        vl_high_resolution_images=True,
        timeout=timeout
    )
    txt = resp.output.choices[0].message.content if resp and resp.output else ""
    if isinstance(txt, list):
        txt = txt[0].get("text", "")
    elif isinstance(txt, dict):
        txt = txt.get("text", "")
    return txt or ""


# === 2.2.2 OpenAI JSON fixing function ===
def fix_bbox_json_with_openai(text: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> Optional[dict]:
    """Call OpenAI to fix the bbox JSON text into valid JSON dict."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it in Run/Debug Configurations.")



    system = (
        "You are a strict JSON fixer. Return ONLY valid, minified JSON with no comments or markdown. "
        "Target schema: {\"bboxes\":[{\"label\":string,\"x1\":int,\"y1\":int,\"x2\":int,\"y2\":int}]}. "
        "Fix key quoting, replace any '=' with ':', remove trailing commas, and drop unknown keys. "
        "If you must guess numeric values, keep them integers. Never add text outside JSON."
    )
    user = (
        "Fix this into valid JSON following the schema exactly. If it already fits, just return it as-is:\n"
        f"{text}"
    )

    try:
        # Apply openai SDK
        resp = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_output_tokens=max_tokens,
        )
        fixed_text = resp.output_text or ""
        # Clean up code block markers if any
        fixed_text = fixed_text.strip()
        if fixed_text.startswith("```"):
            fixed_text = fixed_text.strip("`")
            # If it starts with "json", remove that too
            if fixed_text.startswith("json"):
                fixed_text = fixed_text[len("json"):].lstrip()

        # Try to parse the JSON
        data = json.loads(fixed_text)
        if isinstance(data, dict) and isinstance(data.get("bboxes"), list):
            return data
        return None
    except Exception as e:
        print(f"⚠️ OpenAI fix failed: {e}")
        return None


# === 2.2.3 Manual fixing function ===
def get_bboxes_with_manual_fix(image_path: str,
                               input_text: str,
                               output_dir: str,
                               image_id: str,
                               cleaned_preview: bool = True) -> dict:
    """
    Only used when the automatic JSON parsing fails.
    """
    # 1) Call Qwen to get the raw bbox text
    raw_text = call_qwen_bbox(image_path, input_text)
    print(f"📦 BBOX response (first try): {raw_text}")

    clean_text = clean_and_fix_bbox_json(raw_text)
    print("📦 Cleaned BBOX JSON:", clean_text)
    # 2) If cleaned_text is empty, we need manual fixing
    dbg_dir = os.path.join(output_dir, "debug_bbox_cleaned")
    os.makedirs(dbg_dir, exist_ok=True)
    cand_path = os.path.join(dbg_dir, f"{image_id}.json")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    # 3)

    print("\n📝 Please mannully fix the bbox JSON file in editor. \n")
    print(f"    {cand_path}")
    print("   After saving the file, return here and press Enter to re-parse it, or enter skip.")

    # 4) ready to parse the edited file
    while True:
        user_in = input("[Waiting for your input] Press Enter to re-parse, or type 'skip' to skip this image: ").strip().lower()
        if user_in == "skip":
            print("⏭️ Skip this image")
            return {}

        # Read the edited file
        try:
            with open(cand_path, "r", encoding="utf-8") as f:
                edited_text = f.read()
        except Exception as e:
            print(f"⚠️ Read error: {e}. Please make sure the file is accessible.")
            continue

        parsed = json.loads(edited_text)
        # parsed = clean_and_fix_bbox_json(edited_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("bboxes"), list):
            print("✅ Successfully parsed the edited JSON.")
            return parsed

        # If still not valid, show error context
        from json import JSONDecodeError
        try:
            # Use json.loads to find the error position
            json.loads(edited_text)
        except JSONDecodeError as e:
            start = max(e.pos - 40, 0)
            end = min(e.pos + 40, len(edited_text))
            snippet = edited_text[start:end]
            print(f"❌ Still have illegal {e.msg} @ line {e.lineno}, col {e.colno}")
            print("Relavent content", repr(snippet))
        except Exception as e:
            print(f"❌ Not JSONDecodeError：{e}")

        print("Please correct the json file in editor and try again, or input 'skip' to skip this image.")


# === 2.2.4 Retry function for bbox parsing ===
def get_bboxes_with_retry(image_path: str, input_text: str, retries: int = 1) -> dict:
    """Call Qwen bbox with retries to get valid JSON."""

    resp_text = call_qwen_bbox(image_path, input_text)
    print(f"📦 BBOX response: {resp_text}")
    parsed = clean_and_fix_bbox_json(resp_text)
    if isinstance(parsed, dict) and parsed.get("bboxes"):
        return parsed

    # retry
    retry_text = (
    """ Please verify your JSON format and ensure it strictly follows the specified structure. **Only respond with JSON, no other text**
        "{\n"
    "  \"bboxes\": [\n"
    "    {\"label\": \"<specific label>\", \"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int},\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    """)

    for i in range(retries):
        print(f"⚠️ JSON parsing failed (bbox). Retrying {i+1}/{retries} ...")
        resp_text = call_qwen_bbox(image_path, retry_text+resp_text)
        print(f"📦 BBOX response (retry {i+1}): {resp_text}")
        parsed = clean_and_fix_bbox_json(resp_text)
        if isinstance(parsed, dict) and parsed.get("bboxes"):
            return parsed

    return {}


# === 2.2.5 Parse JSON function ===
def try_parse_json(s: str):
    if not s:
        return None
    t = s.strip()
    # clean up common JSON issues
    t = re.sub(r',(\s*[}\]])', r'\1', t)
    t = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*):', r'\1"\2"\3:', t)
    t = re.sub(r"(?<!\\)'", '"', t)
    try:
        return json.loads(t)
    except Exception:
        return None


# === 2.2.6 Clean and fix bbox JSON function ===
def clean_and_fix_bbox_json(text: str) -> str:
    if not text:
        return {}

    # 1) Remove code block markers
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)

    # 2) Find the JSON block
    js = extract_json_block(text)

    if not js:
        return {}

    # 3) Fix common JSON formatting issues, e.g.:
    #   Transform "key", "value" -> "key": "value"
    js = re.sub(r'"(\w+)"\s*,\s*"(-?\d+\.?\d*)"', r'"\1": \2', js)
    #   Transform "key", "value" -> "key": "value"
    js = re.sub(r'"(\w+)"\s*,\s*"([^"]*?)"', r'"\1": "\2"', js)
    js = re.sub(r'(?<!")\b(x1|y1|x2|y2)\b\s*=\s*', r'"\1": ', js)
    js = re.sub(r'(?<!")\b(x1|y1|x2|y2)\b\s*:\s*', r'"\1": ', js)

    # 4) Remove trailing commas before closing brackets
    js = re.sub(r',(\s*[}\]])', r'\1', js)

    return js
    # # 5) Parse the JSON
    # data = try_parse_json(js)
    # if not isinstance(data, dict):
    #     print("⚠️ Failed to parse JSON after cleaning.\n")
    #     print("The cleaned JSON was:\n", js)
    #     try:
    #         log_path = os.path.join(output_dir, "failure_log.txt")
    #         with open(log_path, "a", encoding="utf-8") as lf:
    #             lf.write(f"{image_id}\tparse_after_cleaning_error\t{e}\n")
    #     except Exception:
    #         pass
    #     return {}
    #
    # # 6) Fix bbox format
    # fixed = []
    # for b in data.get("bboxes", []):
    #     if not isinstance(b, dict):
    #         continue
    #     b = dict(b)  # Copy
    #
    #     # if the keys end with '=', remove the '='
    #     for _k in list(b.keys()):
    #         if isinstance(_k, str) and _k.endswith('='):
    #             b[_k[:-1]] = b.pop(_k)
    #         if isinstance(_k, str) and _k.endswith(':'):
    #             b[_k[:-1]] = b.pop(_k)
    #
    #     # Try to convert numeric values to float
    #     for k in ["x1", "y1", "x2", "y2", "width", "height", "w", "h", "x", "y"]:
    #         if k in b and isinstance(b[k], str):
    #             nums = re.findall(r'-?\d+\.?\d*', b[k])
    #             b[k] = float(nums[0]) if nums else b[k]
    #
    #     # If "x2" or "y2" is missing, calculate from "x1", "y1" and "width", "height"
    #     if "x2" not in b and "width" in b and "x1" in b:
    #         try:
    #             b["x2"] = float(b["x1"]) + float(b.get("width", 0))
    #         except Exception:
    #             pass
    #     if "y2" not in b and "height" in b and "y1" in b:
    #         try:
    #             b["y2"] = float(b["y1"]) + float(b.get("height", 0))
    #         except Exception:
    #             pass
    #
    #     fixed.append(b)
    # return {"bboxes": fixed}


# </editor-fold>


def process_image(image_file: str):
    # (1) Check if the file is an image
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        return None

    image_id = os.path.splitext(image_file)[0].split("_")[0]
    # (2) Check if the image has been processed before
    if image_id in done_images:
        print(f"🔄 Skip {image_file} (cached)")
        return None

    image_path = os.path.join(image_dir, image_file)
    local_coco = []
    # (3) Prepare local COCO and step prediction structures
    local_step = {
        "image_id": image_id,
        "step_top1": "Unknown",
        "step_probs": {k: 1/3 for k in STEP_LABELS},
        "note": "error_or_skip"
    }

    try:
        print(f"\n🔍 Processing {image_file}...")
        # === MAIN PROCESSING STEPS ===
        # === 0) Read image
        try:
            with open(image_path, "rb") as f:
                _ = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Cannot read image: {e}")
            with open(os.path.join(output_dir, "failure_log.txt"), "a", encoding="utf-8") as lf:
                lf.write(f"{image_id}\tread_image_error\t{e}\n")
            # Return placeholder
            return image_id, local_coco, local_step

        # === 1) BBOX Prediction===
        bboxes = []
        try:
            parsed = fix_bbox_json_with_openai(
                call_qwen_bbox(image_path, input_text, temperature=0.0, timeout=90),
                max_tokens=2000,
                model="gpt-4o-mini"
            )
            if isinstance(parsed, dict) and isinstance(parsed.get("bboxes"), list):
                bboxes = parsed["bboxes"]
            else:
                print("⚠️ No valid 'bboxes'; skipping boxes.")
        except Exception as e:
            print(f"⚠️ BBOX error: {e}")
            with open(os.path.join(output_dir, "failure_log.txt"), "a", encoding="utf-8") as lf:
                lf.write(f"{image_id}\tBBOX_pred_error\t{e}\n")

        # === 2) RAG Step Prediction ===
        try:
            description_prompt = (
                "You are an expert in hysterectomy.\n"
                "Describe instruments, tissues, their pose, and the current phase."
            )
            resp = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=[{"role":"user","content":[{"image":image_path},{"text":description_prompt}]}],
                temperature=0.4,
                result_format="message",
                vl_high_resolution_images=True,
                timeout=90
            )
            desc = resp.output.choices[0].message.content if resp and resp.output else ""
            if isinstance(desc, list):
                desc = desc[0].get("text","")
            elif isinstance(desc, dict):
                desc = desc.get("text","")
            step_name = retrieve_step_by_rag(desc.strip()) or "Unknown"
            # one-hot
            step_probs = {k: 0.0 for k in STEP_LABELS}
            if step_name in step_probs:
                step_probs[step_name] = 1.0
            local_step = {
                "image_id": image_id,
                "step_top1": step_name,
                "step_probs": step_probs
            }
        except Exception as e:
            print(f"⚠️ Step/RAG error: {e}")
            with open(os.path.join(output_dir, "failure_log.txt"), "a", encoding="utf-8") as lf:
                lf.write(f"{image_id}\tStep_in_RAG_error\t{e}\n")

        # === 3) Visualization and COCO conversion ===
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
            font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)

            def parse_coord(val):
                if isinstance(val, (int, float)): return int(val)
                if isinstance(val, list) and val:
                    return int(float(val[0])) if str(val[0]).replace('.','',1).isdigit() else 0
                if isinstance(val, str):
                    m = re.search(r'-?\d+\.?\d*', val)
                    return int(float(m.group())) if m else 0
                return 0

            for b in bboxes:
                x1, y1 = parse_coord(b.get("x1")), parse_coord(b.get("y1"))
                x2, y2 = parse_coord(b.get("x2")), parse_coord(b.get("y2"))
                cat_id = label_to_category_id(b.get("label",""))
                if cat_id is not None:
                    local_coco.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "category_name": b["label"],
                        "bbox": xyxy_to_xywh(x1, y1, x2, y2),
                        "score": safe_float(b.get("score",1.0))
                    })
                color = "green" if any(t in b["label"].lower() for t in ["uterus","ovaries","tubes","bladder","ureter"]) else "red"
                draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
                draw.text((x1, max(y1-12,0)), b.get("label",""), fill=color, font=font_def)

            draw.text((50,20), f"Step: {local_step['step_top1']}", fill="blue", font=font_lar)
            img.save(os.path.join(output_dir, f"annotated_{image_file}"))
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
            with open(os.path.join(output_dir, "failure_log.txt"), "a", encoding="utf-8") as lf:
                lf.write(f"{image_id}\tVisualization_error\t{e}\n")

        # (4) Finalize and return results
        return image_id, local_coco, local_step

    except Exception as e:
        # Deal with unexpected errors
        print(f"❌ Unexpected error for {image_file}: {e}")
        with open(os.path.join(output_dir, "failure_log.txt"), "a", encoding="utf-8") as lf:
            lf.write(f"{image_id}\touter_try_except\t{e}\n")
        # Return placeholder
        return image_id, local_coco, local_step


# === Main processing loop with ThreadPoolExecutor ===
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(process_image, img): img
        for img in os.listdir(image_dir)
    }
    for fut in concurrent.futures.as_completed(futures):
        res = fut.result()
        if not res:
            continue
        image_id, coco_list, step_dict = res

        # (1) Update global lists
        coco_detections.extend(coco_list)
        step_predictions.append(step_dict)

        # (2) Write outputs to JSON files
        with open(detections_path, "w", encoding="utf-8") as f:
            json.dump(coco_detections, f, ensure_ascii=False, indent=2)
        with open(steps_path, "w", encoding="utf-8") as f:
            json.dump(step_predictions, f, ensure_ascii=False, indent=2)

        # (3) Cache the processed image
        with open(done_images_file, "a", encoding="utf-8") as f:
            f.write(image_id + "\n")

        print(f"💾 Done {image_id}: outputs updated, cached.")


# === After processing all images: write outputs ===
try:
    with open(steps_path, "w", encoding="utf-8") as f:
        json.dump(step_predictions, f, ensure_ascii=False, indent=2)
    print(f"💾 Step predictions written to: {steps_path}")
except Exception as e:
    print(f"⚠️ Failed to write steps JSON: {e}")
    try:
        log_path = os.path.join(output_dir, "failure_log.txt")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{image_id}\tSteps_JSON_writing_error\t{e}\n")
    except Exception:
        pass
