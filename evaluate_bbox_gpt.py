# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from collections import defaultdict

# ========== Configuration ==========
gt_path = "/home/hiuching-g/PRHK/Ground_Truth/result.json"   # COCO formate json from Label Studio
raw_dir = "/home/hiuching-g/PRHK/Output/Output_GPT5_Benchmark_328/_raw"
output_csv = "/home/hiuching-g/PRHK/Evaluation/eval_bbox_detailed_GPT5.csv"
output_summary = "/home/hiuching-g/PRHK/Evaluation/eval_bbox_summary_GPT5.txt"

IOU_THRESH = 0.5  # IoU threshold for matching predictions to ground truth

# ========== Helper functions==========
def strip_prefix(fname):
    """remove the prefix from the filename"""
    return "_".join(fname.split("_")[1:])

def iou(boxA, boxB):
    """Cal culate the IoU of two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# ========== Load Ground Truth ==========
with open(gt_path, "r", encoding="utf-8") as f:
    gt = json.load(f)

images = {img["id"]: os.path.basename(img["file_name"]) for img in gt["images"]}
categories = {cat["id"]: cat["name"] for cat in gt["categories"]}

# Transfer to dict[filename] = [ (label, [x1,y1,x2,y2]) ... ]
gt_dict = defaultdict(list)
for ann in gt["annotations"]:
    img_name = strip_prefix(images[ann["image_id"]])
    x, y, w, h = ann["bbox"]
    gt_dict[img_name].append((categories[ann["category_id"]], [x, y, x + w, y + h]))

# ========== Evaluate Predictions ==========
records = []

for raw_file in os.listdir(raw_dir):
    if not raw_file.endswith("_raw.txt"):
        continue

    pred_name = raw_file.replace("_raw.txt", ".png")
    stripped_name = strip_prefix(pred_name)

    # Read predictions
    try:
        with open(os.path.join(raw_dir, raw_file), "r", encoding="utf-8") as f:
            pred_data = json.load(f)
            pred_bboxes = pred_data.get("bboxes", [])
    except Exception:
        continue

    gt_bboxes = gt_dict.get(stripped_name, [])
    used = set()

    # match the bbox with GT
    for pb in pred_bboxes:
        plabel = pb["label"]
        pbox = [pb["x1"], pb["y1"], pb["x2"], pb["y2"]]

        best_iou, best_gt = 0, None
        for gi, (glabel, gbox) in enumerate(gt_bboxes):
            if gi in used:
                continue
            if glabel != plabel:
                continue
            i = iou(pbox, gbox)
            if i > best_iou:
                best_iou, best_gt = i, gi

        if best_iou >= IOU_THRESH and best_gt is not None:
            # TP
            used.add(best_gt)
            records.append({"image": stripped_name, "label": plabel, "result": "TP", "iou": best_iou})
        else:
            # FP
            records.append({"image": stripped_name, "label": plabel, "result": "FP", "iou": best_iou})

    # the remaining GT → FN
    for gi, (glabel, gbox) in enumerate(gt_bboxes):
        if gi not in used:
            records.append({"image": stripped_name, "label": glabel, "result": "FN", "iou": 0})

# ========== Summary ==========
df = pd.DataFrame(records)
report = []

for label in sorted(df["label"].unique()):
    sub = df[df["label"] == label]
    tp = (sub["result"] == "TP").sum()
    fp = (sub["result"] == "FP").sum()
    fn = (sub["result"] == "FN").sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    report.append({"label": label, "TP": tp, "FP": fp, "FN": fn,
                   "precision": precision, "recall": recall, "f1": f1})

report_df = pd.DataFrame(report)

# overall micro metrics
tp_total = (df["result"] == "TP").sum()
fp_total = (df["result"] == "FP").sum()
fn_total = (df["result"] == "FN").sum()
precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

# Save the detailed results to CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

# save summary
with open(output_summary, "w", encoding="utf-8") as f:
    f.write("===== Bounding Box Evaluation Summary =====\n")
    f.write(f"Total TP: {tp_total}, FP: {fp_total}, FN: {fn_total}\n")
    f.write(f"Overall Precision: {precision_micro:.3f}\n")
    f.write(f"Overall Recall:    {recall_micro:.3f}\n")
    f.write(f"Overall F1:        {f1_micro:.3f}\n\n")
    f.write("Per-class results:\n")
    f.write(report_df.to_string(index=False))

print("The detailed results has been saved to ", output_csv)
print("The summary has been saved to", output_summary)
