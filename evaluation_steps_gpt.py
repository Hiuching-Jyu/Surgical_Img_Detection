import pandas as pd
import json
import os

# === Document paths ===
steps_path = "/home/hiuching-g/PRHK/Ground_Truth/steps.csv"
result_path = "/home/hiuching-g/PRHK/Ground_Truth/result.json"
raw_dir = "/home/hiuching-g/PRHK/Output/Output_GPT5_Benchmark_328/_raw"   # 存放 *_raw.txt 的目录
output_csv_path = "/home/hiuching-g/PRHK/Evaluation/evaluation_from_raw_GPT5.csv"

# === Load ground truth ===
steps_df = pd.read_csv(steps_path)

# Extract basename（000_surgery01_1005.png）
steps_df["basename"] = steps_df["image"].apply(lambda x: os.path.basename(str(x)))

# Remove the prefix and save it as surgery01_1005.png
def strip_prefix(fname):
    return "_".join(fname.split("_")[1:])

steps_df["stripped"] = steps_df["basename"].apply(strip_prefix)

print("✅ Ground truth columns:", steps_df.columns[:10])
print("Eample stripped:", steps_df["stripped"].head().tolist())


# === Collect the evaluatiosn results ===
records = []

for raw_file in os.listdir(raw_dir):
    if not raw_file.endswith("_raw.txt"):
        continue

    # The corresponding image name
    base_name = raw_file.replace("_raw.txt", ".png")
    stripped_name = strip_prefix(base_name)

    # Read the prediction
    with open(os.path.join(raw_dir, raw_file), "r", encoding="utf-8") as f:
        try:
            pred_data = json.load(f)
            pred_step = pred_data.get("step", None)
        except Exception:
            pred_step = None

    # Find ground truth
    gt_row = steps_df[steps_df["stripped"] == stripped_name]
    if gt_row.empty:
        continue

    gt_step = gt_row["step"].values[0]

    records.append({
        "image_name": stripped_name,
        "gt_step": gt_step,
        "pred_step": pred_step,
        "correct": gt_step == pred_step if pred_step else False
    })

# === Save and output ===
eval_df = pd.DataFrame(records)
accuracy = eval_df["correct"].mean() if not eval_df.empty else 0.0
print(f"✅ Accuracy = {accuracy:.2%}, Total sample = {len(eval_df)}")

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
eval_df.to_csv(output_csv_path, index=False)
print(f"The results have been saved to {output_csv_path}")
