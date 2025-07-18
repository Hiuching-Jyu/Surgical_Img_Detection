import dashscope
from dashscope import MultiModalConversation
import os
import re
import json
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import base64

dashscope.api_key = 'sk-8694ac696f1c42aba2f1cfb254a5918d'

image_dir = "/home/hiuching-g/PRHK/test_images"
output_dir = "/home/hiuching-g/PRHK/Output_QWen_steps"


# upload image and ask for bboxes, and step
input_text = (
    "You are given a robotics surgical image. Perform the following tasks:\n\n"
    "1. Identify and localize all **surgical instruments** in the image, with only its specific name.\n"
    "2. Identify and localize all **body tissue** in the image, with only its specific name.\n"
    "3. Determine the most likely **surgical step** from the following 9 steps:\n"
    "- Port Placement and Docking\n"
    "- Exposure and Inspection\n"
    "- Uterine Mobilization\n"
    "- Vessel Control\n"
    "- Bladder Dissection and Bladder Flap Creation\n"
    "- Colpotomy\n"
    "- Uterus Removal\n"
    "- Vaginal Cuff Closure\n"
    "- Final Hemostasis and Robot Exit\n\n"
    "Output the result in this JSON format **(and only output this JSON)**:\n"
    "{\n"
    "  \"step\": \"<surgical step name>\",\n"
    "  \"bboxes\": [\n"
    "    {\"label\": \"<instrument_name or body_tissue>\", \"x1\": int, \"y1\": int, \"x2\": int, \"y2\": int},\n"
    "    ...\n"
    "  ]\n"
    "}"
)


for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, image_file)
    print(f"\n🔍 Processing {image_file}...")

    # Read local images and transform it to base64 style
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {"role": "user", "content": [{"image": image_path}, {"text": input_text}]}
    ]

    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages,
        temperature=0.3,
        result_format="message",
        vl_high_resolution_images=True
    )
    if not response or not hasattr(response, "output") or not response.output:
        print("DashScope API call failed or returned no output.")
        print("Full response:", response)
        print("Skiping this image")
        continue

    response_text = response.output.choices[0].message.content

    if isinstance(response_text, list):
        response_text = response_text[0].get("text", "")
    elif isinstance(response_text, dict):
        response_text = response_text.get("text", "")
    print("📦 Bbox response:\n", response_text)

    # Extract json from response
    json_text_match = re.search(r'\{[\s\S]+\}', response_text)
    if json_text_match:
        json_text = json_text_match.group(0)
        # Fix wrong output like "x1=" -> "x1"
        json_text = json_text.replace('"x1="', '"x1"')
        json_text = json_text.replace('"x2="', '"x2"')
        json_text = json_text.replace('"y1="', '"y1"')
        json_text = json_text.replace('"y2="', '"y2"')
        try:
            parsed = json.loads(json_text)
            step_name = parsed.get("step", "Unknown Step")
            print(f"Step name: {step_name}")
            bboxes = parsed.get("bboxes", [])
            print(f"Bounding boxes: {bboxes}")
        except json.JSONDecodeError as e:
            print("⚠️ JSON decode failed:", e)
            step_name, bboxes = "Unknown Step", []
    else:
        print("⚠️ No JSON found.")
        step_name, bboxes = "Unknown Step", []

    # 🖼️ Draw the bbox and name
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_def = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=26)
    font_lar = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=36)


    # Draw bboxes
    for box in bboxes:
        try:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = box.get("label", "Tool")

            # Set different colors for tissues and instruments
            if "tissue" in label.lower():
                color = "green"
            else:
                color = "red"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(y1 - 12, 0)), label, fill=color, font=font_def)
        except Exception as e:
            print(f"⚠️ Failed to draw box: {box}, Error: {e}")

    print("✅ Bbox drawn")

    # Write step name
    draw.text((50, 20), f"Step: {step_name}", fill="blue", font=font_lar)
    print("✅ Step written down")

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill="yellow")

    # Save image
    output_path = os.path.join(output_dir, f"annotated_{image_file}")
    img.save(output_path)
    print(f"✅ Saved image with boxes to {output_path}")
