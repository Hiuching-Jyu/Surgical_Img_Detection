import pandas as pd
import json
import os
import re

# === 文件路径 ===
gt_csv_path = "/home/hiuching-g/PRHK/Ground_Truth/steps.csv"
pred_json_path = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_withoutRAG_236/steps_prediction.json"
output_txt_path = "/home/hiuching-g/PRHK/Evaluation/QWen_steps_withoutRAG_236/step_eval.txt"
output_csv_path = "/home/hiuching-g/PRHK/Evaluation/QWen_steps_withoutRAG_236/step_eval_detailed.csv"

# === 读取 GT 文件 ===
df_gt = pd.read_csv(gt_csv_path)

# 固定使用你提供的列名
image_col = "image"
step_col = "step"

# === 提取 xxx_surgery 作为 GT image_id ===
gt_dict = {}
for _, row in df_gt.iterrows():
    full_path = str(row[image_col])  # e.g., /data/upload/1/d5d1738d-000_surgery01_1005.png
    filename = os.path.basename(full_path)
    match = re.search(r'(\d+)_surgery', filename)
    if match:
        image_id = match.group(1).zfill(3)  # 提取 '000' → '000'
        gt_dict[image_id] = str(row[step_col]).strip()

# === 读取预测 JSON 文件 ===
with open(pred_json_path, "r", encoding="utf-8") as f:
    predictions = json.load(f)

# === 计算准确率并写入报告 ===
results_txt = []
results_csv = []
correct = 0
total = 0

for pred in predictions:
    image_id = str(pred["image_id"]).zfill(3)  # '0' → '000'
    pred_step = pred["step_top1"].strip()
    gt_step = gt_dict.get(image_id, "Unknown")

    is_correct = (gt_step == pred_step)
    if is_correct:
        correct += 1
    total += 1

    results_txt.append(f"Image ID: {image_id} | GT: {gt_step} | Pred: {pred_step} | {'✅' if is_correct else '❌'}")
    results_csv.append({
        "image_id": image_id,
        "gt_step": gt_step,
        "pred_step": pred_step,
        "correct": is_correct
    })

accuracy = correct / total if total > 0 else 0.0
results_txt.append(f"\nTotal: {total}")
results_txt.append(f"Correct: {correct}")
results_txt.append(f"Accuracy: {accuracy:.2%}")

# === 写入 TXT ===
os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_txt))

# === 写入 CSV ===
df_results = pd.DataFrame(results_csv)
df_results.to_csv(output_csv_path, index=False)

print(f"✅ Evaluation done. Accuracy: {accuracy:.2%}")
print(f"📄 TXT Results saved to: {output_txt_path}")
print(f"📊 CSV Results saved to: {output_csv_path}")
