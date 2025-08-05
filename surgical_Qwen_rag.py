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
os.environ["OPENAI_API_KEY"] = 
# <editor-fold desc=" 1. Initialization for API-Key and dirs">
dashscope.api_key = 'sk-8694ac696f1c42aba2f1cfb254a5918d'  # Replace with your key

image_dir = "/home/hiuching-g/PRHK/test_images_236"
output_dir = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_RAG_236"
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


# === helpers ===
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



def fix_bbox_json_with_openai(text: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> Optional[dict]:
    """Call OpenAI to fix the bbox JSON text into valid JSON dict."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it in Run/Debug Configurations.")

    _openai_client = OpenAI()

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
        resp = _openai_client.responses.create(
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


# collectors
coco_detections = []
step_predictions = []

# </editor-fold>

for image_file in os.listdir(image_dir):
    try:
        # Only process image files
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, image_file)
        # image_id = image_file  # use full filename as ID
        image_id = os.path.splitext(image_file)[0].split("_")[0]
        print(f"\n🔍 Processing {image_file}...")

        # Try to read the image and convert to base64
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Cannot read image {image_path}: {e}. Skipping.")
            try:
                log_path = os.path.join(output_dir, "failure_log.txt")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{image_id}\tread_image_error\t{e}\n")
            except Exception:
                pass
            # Also write a placeholder step prediction
            step_predictions.append({
                "image_id": image_id,
                "step_top1": "Unknown",
                "step_probs": {
                    "Preparation & Exposure": 1 / 3,
                    "Dissection & Vessel Control": 1 / 3,
                    "Uterus Removal & Closure": 1 / 3
                },
                "note": "image_read_fail"
            })
            continue

        # === 1) predict bbox ===
        bboxes = []
        try:
            parsed = fix_bbox_json_with_openai(
                call_qwen_bbox(image_path, input_text, temperature=0.4, timeout=90),
                max_tokens=2000,
                model="gpt-4o-mini"
            )
            print("📦 Parsed BBOX JSON:", parsed)
            if isinstance(parsed, dict) and isinstance(parsed.get("bboxes"), list):
                bboxes = parsed["bboxes"]
            else:
                # Optional: fallback to manual fixing
                print("⚠️ No valid 'bboxes' list parsed; continuing without boxes.")

        except Exception as e:
            print(f"⚠️ BBOX prediction error: {e}. Continue without boxes.")
            try:
                log_path = os.path.join(output_dir, "failure_log.txt")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{image_id}\tBBOX_pred_error\t{e}\n")
            except Exception:
                pass


        # === 2) RAG step prediction ===
        step_name = "Unknown"
        try:
            description_prompt = (
                "You are an expert in hysterectomy.\n"
                "Please describe the following image, including:\n"
                "- instruments name\n"
                "- body tissue name\n"
                "- their location and orientation\n"
                "- and the current surgical status or phase."
            )
            messages_desc = [{"role": "user", "content": [{"image": image_path}, {"text": description_prompt}]}]
            response_desc = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages_desc,
                temperature=0.4,
                result_format="message",
                vl_high_resolution_images=True,
                timeout=90
            )
            description_text = response_desc.output.choices[
                0].message.content if response_desc and response_desc.output else ""
            if isinstance(description_text, list):
                description_text = description_text[0].get("text", "")
            elif isinstance(description_text, dict):
                description_text = description_text.get("text", "")
            print("Description for RAG printed\n")

            # use RAG to retrieve the step name
            step_name = "Unknown"
            step_name = retrieve_step_by_rag((description_text or "").strip()) or "Unknown"
        except Exception as e:
            print(f"⚠️ Step (RAG) error: {e}. Use 'Unknown'.")
            try:
                log_path = os.path.join(output_dir, "failure_log.txt")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{image_id}\tStep_in_RAG_error\t{e}\n")
            except Exception:
                pass

        # one-hot encode the step name
        step_probs = {k: 0.0 for k in STEP_LABELS}
        if step_name in step_probs:
            step_probs[step_name] = 1.0
        step_predictions.append({
            "image_id": image_id,
            "step_top1": step_name,
            "step_probs": step_probs
        })

        # === 3) Visualization and COCO format conversion ===
        try:
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
            font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)


            def parse_coord(val):
                if isinstance(val, int): return val
                if isinstance(val, float): return int(val)
                if isinstance(val, list) and val:
                    try:
                        return int(float(val[0]))
                    except Exception:
                        return 0
                if isinstance(val, str):
                    nums = re.findall(r'-?\d+\.?\d*', val)
                    try:
                        return int(float(nums[0])) if nums else 0
                    except Exception:
                        return 0
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
                        try:
                            log_path = os.path.join(output_dir, "failure_log.txt")
                            with open(log_path, "a", encoding="utf-8") as lf:
                                lf.write(f"{image_id}\tunknown_label_error\t{e}\n")
                        except Exception:
                            pass
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
                    try:
                        log_path = os.path.join(output_dir, "failure_log.txt")
                        with open(log_path, "a", encoding="utf-8") as lf:
                            lf.write(f"{image_id}\tbox_drawing_error\t{e}\n")
                    except Exception:
                        pass

            draw.text((50, 20), f"Step: {step_name}", fill="blue", font=font_lar)
            output_path = os.path.join(output_dir, f"annotated_{image_file}")
            try:
                img.save(output_path)
                print(f"✅ Annotated image saved to: {output_path}")
            except Exception as e:
                print(f"⚠️ Failed to save annotated image for {image_file}: {e}")
                try:
                    log_path = os.path.join(output_dir, "failure_log.txt")
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"{image_id}\tannotated_image_saving_error\t{e}\n")
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Visualization error on {image_file}: {e}. Continue.")
            try:
                log_path = os.path.join(output_dir, "failure_log.txt")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{image_id}\tVisualization_error\t{e}\n")
            except Exception:
                pass

    except Exception as e:
        # outer try-except to catch any unexpected errors
        print(f"❌ Unexpected error for {image_file}: {e}. Skipping this image.")
        try:
            log_path = os.path.join(output_dir, "failure_log.txt")
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"{image_id}\tStep_pred_error\t{e}\n")
        except Exception:
            pass
        # write a placeholder step prediction
        image_id = os.path.splitext(image_file)[0].split("_")[0]
        step_predictions.append({
            "image_id": image_id,
            "step_top1": "Unknown",
            "step_probs": {
                "Preparation & Exposure": 1 / 3,
                "Dissection & Vessel Control": 1 / 3,
                "Uterus Removal & Closure": 1 / 3
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
        log_path = os.path.join(output_dir, "failure_log.txt")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{image_id}\tDetection_JSON_writing_error\t{e}\n")
    except Exception:
        pass

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
