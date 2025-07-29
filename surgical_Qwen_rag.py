import dashscope
from dashscope import MultiModalConversation
import os
import re
import json
from PIL import Image, ImageDraw, ImageFont
import base64
from rag_module import retrieve_step_by_rag

# <editor-fold desc=" 1. Initialization for API-Key and dirs">
dashscope.api_key = 'sk-8694ac696f1c42aba2f1cfb254a5918d'  # Replace with your key

image_dir = "/home/hiuching-g/PRHK/test_images"
output_dir = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_RAG"

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

# </editor-fold>


for image_file in os.listdir(image_dir):
    # <editor-fold desc=" 2. Use bbox_prompts for Qwen to get bboxes and labels. ">
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, image_file)
    print(f"\n🔍 Processing {image_file}...")

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    # 🧩 Pipeline 1: Get bounding boxes
    messages_bbox = [{"role": "user", "content": [{"image": image_path}, {"text": input_text}]}]

    response_bbox = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages_bbox,
        temperature=0.4,
        result_format="message",
        vl_high_resolution_images=True
    )

    response_text = response_bbox.output.choices[0].message.content if response_bbox and response_bbox.output else ""
    if isinstance(response_text, list):
        response_text = response_text[0].get("text", "")
    elif isinstance(response_text, dict):
        response_text = response_text.get("text", "")

    json_match = re.search(r'\{[\s\S]+\}', response_text)
    bboxes, all_labels = [], []
    if json_match:
        json_str = json_match.group(0).replace('"x1="', '"x1"').replace('"x2="', '"x2"').replace('"y1="', '"y1"').replace('"y2="', '"y2"')
        try:
            parsed = json.loads(json_str)
            bboxes = parsed.get("bboxes", [])
            all_labels = [box["label"] for box in bboxes if "label" in box]
        except json.JSONDecodeError as e:
            print("⚠️ JSON parsing failed:", e)

    # </editor-fold>

    # <editor-fold desc="3. Use descriptive prompt for qwen to predict steps with RAG">
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
        vl_high_resolution_images=True
    )

    description_text = response_desc.output.choices[0].message.content if response_desc and response_desc.output else ""
    if isinstance(description_text, list):
        description_text = description_text[0].get("text", "")
    elif isinstance(description_text, dict):
        description_text = description_text.get("text", "")
    print("📝 Description for RAG:\n", description_text.strip())

    # Use RAG with free-text description
    step_name = retrieve_step_by_rag(description_text.strip())
    print(f"🔍 RAG Step Inferred: {step_name}")

    # </editor-fold>

    # <editor-fold desc="4. Draw bounding boxes and step name on image">
    # Draw image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)

    # Draw bounding boxes
    def parse_coord(val):
        if isinstance(val, int): return val
        if isinstance(val, list) and val: return int(val[0])
        if isinstance(val, str):
            nums = re.findall(r'\d+', val)
            return int(nums[0]) if nums else 0
        return 0

    for box in bboxes:
        try:
            x1 = parse_coord(box.get("x1"))
            y1 = parse_coord(box.get("y1"))
            x2 = parse_coord(box.get("x2"))
            y2 = parse_coord(box.get("y2"))
            label = box.get("label", "Unknown")
            color = "green" if any(k in label.lower() for k in ["uterus", "ovaries", "tubes", "bladder", "ureter"]) else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(y1 - 12, 0)), label, fill=color, font=font_def)
        except Exception as e:
            print(f"⚠️ Failed to draw box: {box}, Error: {e}")

    # Write step name
    draw.text((50, 20), f"Step: {step_name}", fill="blue", font=font_lar)
    output_path = os.path.join(output_dir, f"annotated_{image_file}")
    img.save(output_path)
    print(f"✅ Annotated image saved to: {output_path}")

    # </editor-fold>





# for image_file in os.listdir(image_dir):
#     if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#         continue
#
#     image_path = os.path.join(image_dir, image_file)
#     print(f"\n🔍 Processing {image_file}...")
#
#     with open(image_path, "rb") as f:
#         base64_image = base64.b64encode(f.read()).decode("utf-8")
#
#     messages = [{"role": "user", "content": [{"image": image_path}, {"text": input_text}]}]
#
#     response = MultiModalConversation.call(
#         model='qwen-vl-plus',
#         messages=messages,
#         temperature=0.4,
#         result_format="message",
#         vl_high_resolution_images=True
#     )
#
#     response_text = response.output.choices[0].message.content if response and response.output else ""
#     if isinstance(response_text, list):
#         response_text = response_text[0].get("text", "")
#     elif isinstance(response_text, dict):
#         response_text = response_text.get("text", "")
#
#     print("Raw response:\n", response_text)
#
#     json_match = re.search(r'\{[\s\S]+\}', response_text)
#     bboxes, all_labels = [], []
#     if json_match:
#         json_str = json_match.group(0).replace('"x1="', '"x1"').replace('"x2="', '"x2"').replace('"y1="', '"y1"').replace('"y2="', '"y2"')
#         try:
#             parsed = json.loads(json_str)
#             bboxes = parsed.get("bboxes", [])
#             all_labels = [box["label"] for box in bboxes]
#         except json.JSONDecodeError as e:
#             print("⚠️ JSON parsing failed:", e)
#
#     # 🧠 Get step from RAG
#     step_name = retrieve_step_by_rag(all_labels)
#     print(f"🔍 RAG Step Inferred: {step_name}")
#
#     # 🖼️ Draw
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)
#     font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
#     font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
#
#     for box in bboxes:
#         try:
#             x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
#             label = box.get("label", "Unknown")
#             color = "green" if any(k in label.lower() for k in ["uterus", "ovaries", "tubes", "bladder", "ureter"]) else "red"
#             draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
#             draw.text((x1, max(y1 - 12, 0)), label, fill=color, font=font_def)
#         except Exception as e:
#             print(f"⚠️ Failed to draw box: {box}, Error: {e}")
#
#     draw.text((50, 20), f"Step: {step_name}", fill="blue", font=font_lar)
#     output_path = os.path.join(output_dir, f"annotated_{image_file}")
#     img.save(output_path)
#     print(f"✅ Annotated image saved to: {output_path}")
