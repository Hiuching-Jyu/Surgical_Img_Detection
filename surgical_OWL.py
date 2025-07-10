import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
# processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")



# Directories
input_dir = "/home/hiuching-g/PRHK/test_images"
output_dir = "/home/hiuching-g/PRHK/Output_OWL_ViT0.2"
os.makedirs(output_dir, exist_ok=True)

# Prompts
prompts = [
    "Grasper", "Scissors", "Needle driver", "Electrocautery hook", "Dissector", "Forceps", "Surgical clip", "Clip applier", "Suction device", "Trocar", "Retractor",
    "Uterus", "Ovary", "Fallopian tube", "Broad ligament", "Bladder", "Abdominal wall", "Fat tissue", "Peritoneum", "Mesentery", "Pelvic cavity",
    "Bleeding", "Blood clot", "Adhesion", "Scar tissue", "Tissue damage", "Inflammation", "Burned tissue"
]

# Font for labels
font = ImageFont.load_default()

# Process each image
for file_name in os.listdir(input_dir):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(input_dir, file_name)
    image = Image.open(image_path).convert("RGB")

    # Inference
    inputs = processor(text=prompts, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=[image.size[::-1]],
        threshold=0.2
    )[0]

    # Draw results
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        box = [int(x) for x in box.tolist()]
        label_name = prompts[label]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 10)), f"{label_name} ({score:.2f})", fill="yellow", font=font)

    # Save result
    # output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_OWL_ViT.png")
    output_path = os.path.join(output_dir, f"OWL_{file_name}")
    image.save(output_path)
    print(f"[✓] Saved: {output_path}")
    torch.cuda.empty_cache()

