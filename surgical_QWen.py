import os
import torch
import json
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
torch.cuda.empty_cache()

# <editor-fold desc=" 1. Load Model and provicde prompts">

# 1.1 Provide prompt
question = """Identify and localise the following objects from the image:

(1) Surgical instruments like graspers, clip appliers, scissors, suction
(2) Relevant anatomic or pathologic targets like tumour mass, bleeding stump, necrotic tissue, mesenteric vessel

Return output in pure JSON array format (no markdown, no explanation, no prefix text). Each JSON object must contain these keys:

- "image_name": string
- "image_index": int
- "label": string
- "x1": int
- "y1": int
- "x2": int
- "y2": int

Coordinates are in absolute pixel values relative to the original image.

Output example:
[
  {"image_name": "[original name of the img].png", "image_index": 0, "label": "Grasper", "x1": ???, "y1": ???, "x2": ???, "y2": ???}
]
"""

# 1.2 Load Model and Processor

model_path = '/home/hiuching-g/PRHK/Qwen'
img_folder = "/home/hiuching-g/PRHK/test_images"
image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, quantization_config=bnb_config, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to("cpu")
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor.save_pretrained(model_path)

# </editor-fold>


for idx, img_file in enumerate(image_files):
    # <editor-fold desc=" 2. Process each image and generate json output">
    img_path = os.path.join(img_folder, img_file)
    print(f"Processing image: {img_file}, size: {Image.open(img_path).size}")

    # 2.1 Input configuration
    image = Image.open(img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    # inputs = inputs.to('cuda')

    # 2.2 Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    # </editor-fold>

    # <editor-fold desc="3. Post-process the JSON file, and save the annotated image">

    # 3.1 Extract the raw JSON string from the output ==
    raw_json = output_text[0]

    raw_json = raw_json.strip()  # delete leading/trailing whitespace
    if raw_json.endswith(","):
        raw_json = raw_json[:-1]  # delete trailing comma
    if not raw_json.endswith("]"):
        raw_json += "]"  # make sure the string ends with a closing bracket

    raw_json = raw_json.replace("[original name of the img].png", img_file)


    # img_path = "/home/hiuching-g/PRHK/test_images/surgery07_2615.png"
    json_path  = "/home/hiuching-g/PRHK/Output_Qwen/qwen.json"
    out_dir    = "/home/hiuching-g/PRHK/Output_Qwen/"
    os.makedirs(out_dir, exist_ok=True)

    # 3.2 Load the JSON data and adjust bounding box coordinates
    data = json.loads(raw_json)  # make sure the JSON is valid
    image = Image.open(img_path)
    print("Original image size", image.size)
    original_width, original_height = image.size
    model_input_size = processor.image_processor.size
    print("processor image size", model_input_size)

    if "width" in model_input_size and "height" in model_input_size:
        model_width = model_input_size["width"]
        model_height = model_input_size["height"]
    elif "shortest_edge" in model_input_size:
        model_width = model_height = model_input_size["shortest_edge"]
    else:
        raise ValueError(f"Unrecognized image size format: {model_input_size}")

    # 3.3 fix bounding box coordinates
    for obj in data:
        obj["x1"] = int(obj["x1"] / original_width * model_width)
        obj["y1"] = int(obj["y1"] / original_height * model_height)
        obj["x2"] = int(obj["x2"] / original_width * model_width)
        obj["y2"] = int(obj["y2"] / original_height * model_height)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 3.4 Annotate the image with bounding boxes and labels
    img = cv2.imread(img_path)
    for obj in data:
        label = obj["label"]
        x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # 3.5 Save the annotated image
    out_path = os.path.join(out_dir, f"Qwen_annotated_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, img)
    print(f"✅ Saved annotated image to: {out_path}")
    open("/home/hiuching-g/PRHK/Output_Qwen/qwen.json", "w").write("")

    # </editor-fold>
