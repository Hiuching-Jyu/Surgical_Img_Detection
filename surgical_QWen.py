from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import json
torch.cuda.empty_cache()
import cv2

# <editor-fold desc="1. Recognition">
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置模型加载GPU（1号GPU）
model_path = '/home/hiuching-g/PRHK/Qwen'  # 修改为本地模型下载地址

# 加载模型
# 根据实际情况修改
img_path = "/home/hiuching-g/PRHK/test_images/surgery07_2615.png"
# 目标定位框信息提问
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
  {"image_name": "[original name of the img].png", "image_index": 0, "label": "Grasper", "x1": 120, "y1": 90, "x2": 180, "y2": 160}
]
"""

# 加载模型
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to("cpu")

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
processor.save_pretrained(model_path)
# 输入配置
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

# 推理
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# </editor-fold>

# <editor-fold desc="2. Post-processing and saving results">
# output_text = ['[\n  {"image_name": "[original name of the img].png", "image_index": 0, "label": "Suction", "x1": 120, "y1": 90, "x2": 180, "y2": 160},\n  {"image_name": "[original name of the img].png", "image_index": 0, "label": "Scissors", "x1": 120, "y1": 90, "x2": 180, "y2": 160},\n ']
# Step 1: 提取字符串
raw_json = output_text[0]

# Step 2: 补上缺失的 JSON 结尾，并移除末尾多余逗号
raw_json = raw_json.strip()  # 去除前后空格和换行
if raw_json.endswith(","):
    raw_json = raw_json[:-1]  # 去掉结尾多余逗号
if not raw_json.endswith("]"):
    raw_json += "]"  # 确保结尾有 ]

# Step 3: 将其中的 [original name of the img] 替换为实际文件名
img_filename = "surgery07_2615.png"  # 根据你的图像路径实际修改
raw_json = raw_json.replace("[original name of the img]", img_filename)

# # Step 4: 尝试解析 JSON
# try:
#     data = json.loads(raw_json)
#     print("✅ JSON 解析成功！")
# except json.JSONDecodeError as e:
#     print("❌ JSON 不合法:", e)

img_path = "/home/hiuching-g/PRHK/test_images/surgery07_2615.png"
json_path  = "/home/hiuching-g/PRHK/Output_Qwen/qwen.json"
out_dir    = "/home/hiuching-g/PRHK/Output_Qwen/"
os.makedirs(out_dir, exist_ok=True)

# Step 1: 将 output_text 写入 json 文件
# clean_json_str = raw_json[0].replace("\n", "").replace("[original name of the img]", os.path.basename(img_path))
data = json.loads(raw_json)  # 确保内容是合法 JSON
image = Image.open(img_path)
print("📐 原图大小:", image.size)
original_width, original_height = image.size  # PIL.Image 原图大小
model_input_size = processor.image_processor.size  # 通常为 dict，例如 {"height": 448, "width": 448}
print("📐 processor 是否改变图像尺寸？", model_input_size)

if "width" in model_input_size and "height" in model_input_size:
    model_width = model_input_size["width"]
    model_height = model_input_size["height"]
elif "shortest_edge" in model_input_size:
    model_width = model_height = model_input_size["shortest_edge"]
else:
    raise ValueError(f"Unrecognized image size format: {model_input_size}")

# 修正每个 box 的位置
for obj in data:
    obj["x1"] = int(obj["x1"] / original_width * model_width)
    obj["y1"] = int(obj["y1"] / original_height * model_height)
    obj["x2"] = int(obj["x2"] / original_width * model_width)
    obj["y2"] = int(obj["y2"] / original_height * model_height)
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

# Step 2: 加载图像并绘制标注
img = cv2.imread(img_path)
for obj in data:
    label = obj["label"]
    x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

# Step 3: 保存输出图像
out_path = os.path.join(out_dir, f"Qwen_annotated_{os.path.basename(img_path)}")
cv2.imwrite(out_path, img)
print(f"✅ Saved annotated image to: {out_path}")

# </editor-fold>