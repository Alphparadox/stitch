import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse
import re

BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_DIR = "/home/naveenkumar/stitch"
MODEL_INPUT_SIZE = 336  # Typical for recent LLaVA models

def get_full_image_path(relative_path):
    path = os.path.join(BASE_IMAGE_DIR, relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    return path

def load_benchmark_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        print("[Image] Path:", image_path, "| Size:", img.size)
        # Check whether image is large enough for reasoning
        if img.size[0] < 200 or img.size[1] < 200:
            print(f"[WARNING] Image {image_path} may be too small for spatial reasoning.")
        # Resize (preserve aspect ratio if needed) to model's input size
        if img.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
            # Use thumbnail to retain full scene (less cropping risk than resize)
            img = img.copy()
            img.thumbnail((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
            bg = Image.new("RGB", (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (255, 255, 255))
            bg.paste(img, ((MODEL_INPUT_SIZE-img.size[0])//2, (MODEL_INPUT_SIZE-img.size[1])//2))
            img = bg
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None

def build_partitioned_option_transformation_prompt():
    return (
        "<image>\n"
        "This puzzle shows a top row with a transformation (e.g., rotation, flip, view change) between left and right images.\n"
        "Each option (A, B, and C) below has two images: a left image (input) and a right image (candidate result after transformation).\n\n"
        "Step 1: Analyze the top row—clearly describe the transformation or rule, including angle and direction if a rotation is shown.\n"
        "Step 2: For each option (A, B, C), describe what transformation occurs from left to right images. "
        "Is it a rotation? What is the angle and direction? Is it a different kind of change?\n"
        "Step 3: Compare the transformation in each option to the one in the top row. State whether it is the same or different.\n"
        "Step 4: Select the option whose transformation most closely matches the top row.\n"
        "End with: The correct option is [A/B/C]."
    )

def extract_answer(raw_text):
    # Robust answer extraction for "The correct option is [A/B/C]"
    pattern = r"The correct option is\s*\[*([ABC])\]*"
    matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    for char in reversed(raw_text.upper()):
        if char in "ABC":
            return char
    return ""

def run_kiva_benchmark(benchmark_data, model, processor):
    print("\n--- LLaVA Visual Analogy Benchmark ---")
    correct = 0
    total = len(benchmark_data)
    for idx, sample in enumerate(benchmark_data, 1):
        print(f"\n[Item {idx}/{total}]")
        try:
            img_path = get_full_image_path(sample["image"])
            img = load_benchmark_image(img_path)
            if img is None:
                print(f"[SKIP] No valid image for item {idx}")
                continue

            prompt = build_partitioned_option_transformation_prompt()
            gt = sample["ground_truth_answer"].strip().upper()

            inputs = processor(
                text=prompt,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)
            print("[Batch] Tensor shape:", inputs['pixel_values'].shape)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=200)
            raw_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            choice = extract_answer(raw_answer)

            print(f"GT = {gt}")
            print(f"Model raw answer:\n{raw_answer}\nChoice: {choice}")

            if choice == gt:
                print(f"CORRECT ✅")
                correct += 1
            else:
                print(f"INCORRECT ❌ Expected: {gt}, Got: {choice or 'None'}")

        except Exception as e:
            print(f"[ERROR] Processing item {idx}: {e}")

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n--- Benchmark Results ---")
    print(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA visual analogy benchmark (partitioned option, full image).")
    parser.add_argument('--benchmark_file', type=str, required=True, help='Path to the benchmark JSON file')
    args = parser.parse_args()

    print("Loading benchmark data...")
    try:
        with open(args.benchmark_file, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items.")
    except Exception as e:
        print(f"[ERROR] Benchmark data loading: {e}")
        return

    print("Loading LLaVA model...")
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, torch_dtype=dtype).to(DEVICE)
        print(f"Model ready on {DEVICE}.")
    except Exception as e:
        print(f"[ERROR] Model loading: {e}")
        return

    run_kiva_benchmark(data, model, processor)

if __name__ == "__main__":
    main()
