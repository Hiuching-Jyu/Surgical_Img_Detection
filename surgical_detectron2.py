import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 1. Load model config and weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # you can change to the other model if you want
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# 2. Directory and prompt setup
INPUT_DIR  = "/home/hiuching-g/PRHK/test_images"
OUTPUT_DIR = "/home/hiuching-g/PRHK/Output_Detectron2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NOTE: Detectron2 does not use text prompts like Qwen, it uses a pre-trained model to detect objects in images.

# 3. Inference and annotation
for file_name in os.listdir(INPUT_DIR):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(INPUT_DIR, file_name)
    image = cv2.imread(img_path)
    outputs = predictor(image)

    # visualize results
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # save annotated image
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file_name)[0] + ".png")
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"Saved annotated image to {output_path}")

