import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os

# ============== CONFIGURE THESE ==============
MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use a direct, known, local path to the sample image
SAMPLE_IMAGE_PATH = "/home/naveenkumar/stitch/2DRotation-90_19_0_-A.jpg"
MODEL_INPUT_SIZE = 336    # or 224 or 448 for some LLaVA versions

def load_image_for_llava(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        print("Original image size:", img.size)
        img = img.copy()
        img.thumbnail((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
        bg = Image.new("RGB", (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (255, 255, 255))
        bg.paste(img, ((MODEL_INPUT_SIZE-img.size[0])//2, (MODEL_INPUT_SIZE-img.size[1])//2))
        img = bg
        print("Resized image size:", img.size)
        return img
    except Exception as e:
        print("[ERROR] Could not open or resize image:", e)
        return None

def main():
    # Load processor and model
    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(DEVICE)

    # Load one test image
    img = load_image_for_llava(SAMPLE_IMAGE_PATH)
    if img is None:
        print("Image load failed, exiting.")
        return

    # Create a debug/describe prompt for vision check
    prompt = (
        "<image>\n"
        "Describe in detail everything you see in this image."
    )

    # Encode input
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    print("Image tensor shape:", inputs['pixel_values'].shape)
    print("Sample tensor stats: min=", inputs['pixel_values'].min().cpu().item(), 
          "max=", inputs['pixel_values'].max().cpu().item(), 
          "mean=", inputs['pixel_values'].mean().cpu().item())

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    raw_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\n=== MODEL OUTPUT ===")
    print(raw_answer)

if __name__ == "__main__":
    main()
