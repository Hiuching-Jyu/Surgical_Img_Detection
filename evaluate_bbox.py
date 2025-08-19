import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
matplotlib.use('TkAgg')


# === Step 1: Load files ===
gt_path = "/home/hiuching-g/PRHK/Ground_Truth/result.json"
dt_path = "/home/hiuching-g/PRHK/Output/Output_QWen_steps_RAG_236/detections_coco.json"
gt_tmp_path = "/home/hiuching-g/PRHK/Ground_Truth/temp_gt_236.json"
dt_tmp_path = "/home/hiuching-g/PRHK/Ground_Truth/temp_dt_236.json"
image_root = "/home/hiuching-g/PRHK/Ground_Truth/images"
save_dir = "/home/hiuching-g/PRHK/Evaluation/QWen_steps_RAG_236"


with open(gt_path, 'r') as f:
    gt_data = json.load(f)

with open(dt_path, 'r') as f:
    dt_data = json.load(f)



# === Step 2: Extract first 236 image_ids from detection file ===
predicted_image_ids = set()
for img in dt_data.get("images", []):
    predicted_image_ids.add(img["id"])
    if len(predicted_image_ids) >= 236:
        break

# === Step 3: Filter GT images and annotations based on predicted image_ids ===
filtered_gt_images = [img for img in gt_data["images"] if img["id"] in predicted_image_ids]
filtered_gt_annotations = [ann for ann in gt_data["annotations"] if ann["image_id"] in predicted_image_ids]


# === Step 4: Add "score" field to predicted annotations if missing ===
for ann in dt_data["annotations"]:
    if "score" not in ann:
        ann["score"] = 1.0

# === Step 5: Save filtered GT and prediction to temp files ===

filtered_gt = {
    "info": gt_data.get("info", {}),
    "licenses": gt_data.get("licenses", []),
    "images": filtered_gt_images,
    "annotations": filtered_gt_annotations,
    "categories": gt_data["categories"]
}

# Filter dt_data annotations to only include those with image_ids in predicted_image_ids
filtered_dt_annotations = [
    ann for ann in dt_data["annotations"]
    if ann["image_id"] in predicted_image_ids
]

# Check if all predicted image_ids are present in GT
dt_image_ids = set(ann["image_id"] for ann in filtered_dt_annotations)
gt_image_ids = set(img["id"] for img in filtered_gt["images"])
missing_ids = dt_image_ids - gt_image_ids
if missing_ids:
    print(f"❌ {len(missing_ids)} predicted image_ids not found in GT:", sorted(list(missing_ids)))
else:
    print("✅ All prediction image_ids match ground truth")

with open(gt_tmp_path, 'w') as f:
    json.dump(filtered_gt, f)

with open(dt_tmp_path, 'w') as f:
    json.dump(filtered_dt_annotations, f)
# === Step 6: Run COCO Evaluation ===
coco_gt = COCO(gt_tmp_path)
coco_dt = coco_gt.loadRes(dt_tmp_path)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print detailed results of TP, FP, FN
for img_id in sorted(predicted_image_ids):
    gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
    dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

    num_gt = len(gt_ann_ids)
    num_pred = len(dt_ann_ids)

    print(f"[Image {img_id}]  GT annotations: {num_gt}, Predictions: {num_pred}")

# Calculate and print precision, recall, and F1-score for each category

def visualize_image(image_id, coco_gt, coco_dt, image_root, save_dir):
    # Retrieve image info from GT
    img_info = coco_gt.loadImgs(image_id)[0]
    file_name = os.path.basename(img_info['file_name'])
    file_path = os.path.join(image_root, file_name)

    # Load  and display the image
    img = Image.open(file_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    # GT bbox
    anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
    for ann in anns_gt:
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1]-5, f"GT:{coco_gt.loadCats([ann['category_id']])[0]['name']}", color='green')

    # Pred bbox
    anns_dt = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=image_id))
    for ann in anns_dt:
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1]+bbox[3]+5, f"Pred:{coco_gt.loadCats([ann['category_id']])[0]['name']} ({ann['score']:.2f})", color='red')

    plt.axis("off")
    plt.title(f"Image ID: {image_id}")
    save_path = os.path.join(save_dir, f"vis_{image_id}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


# Visualize and save all the images
for img_id in predicted_image_ids:
    visualize_image(img_id, coco_gt, coco_dt, image_root, save_dir)
