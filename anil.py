import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse
import re

BASE_MODEL_PATH = "/home/naveenkumar/load/llava-v1.6-vicuna-13b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_DIR = "/home/naveenkumar/stitch"
MODEL_INPUT_SIZE = 336  # Standard for recent LLaVA models

def get_full_image_path(relative_path):
    path = os.path.join(BASE_IMAGE_DIR, relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    return path

def load_benchmark_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        print("[Image] Path:", image_path, "| Size:", img.size)
        if img.size[0] < 200 or img.size[1] < 200:
            print(f"[WARNING] Image {image_path} may be too small for spatial reasoning.")
        # Resize/canvas for exact input size, avoid cropping content
        if img.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
            img = img.copy()
            img.thumbnail((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
            bg = Image.new("RGB", (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (255, 255, 255))
            bg.paste(img, ((MODEL_INPUT_SIZE-img.size[0])//2, (MODEL_INPUT_SIZE-img.size[1])//2))
            img = bg
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None

def build_observational_prompt():
    return (
        "<image>\n"
        "Look carefully at the images. What transformation happens from left to right in the top row? "
        "For each option (A, B, and C) below: does the change from left to right within the option match the top row's transformation? "
        "Which option best matches? Only write: The correct option is [A/B/C]."
    )

def build_visual_debug_prompt():
    return (
        "<image>\n"
        "Describe every object, animal, and transformation shown anywhere in this image. Be as detailed as possible."
    )

def extract_answer(raw_text):
    pattern = r"The correct option is\s*\[*([ABC])\]*"
    matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    for char in reversed(raw_text.upper()):
        if char in "ABC":
            return char
    return ""

def run_kiva_benchmark(benchmark_data, model, processor, debug=False):
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

            prompt = build_visual_debug_prompt() if debug else build_observational_prompt()
            gt = sample.get("ground_truth_answer", "").strip().upper()

            inputs = processor(
                text=prompt,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)
            print("[Batch] Tensor shape:", inputs['pixel_values'].shape)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=250 if debug else 150)
            raw_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            choice = extract_answer(raw_answer) if not debug else None

            print(f"GT = {gt}" if gt else "")
            print(f"Model raw answer:\n{raw_answer}")
            if not debug:
                print(f"Extracted Choice: {choice}")
                if choice == gt:
                    print(f"CORRECT ✅")
                    correct += 1
                else:
                    print(f"INCORRECT ❌ Expected: {gt}, Got: {choice or 'None'}")
            else:
                print("Debug mode: no answer extracted.")

        except Exception as e:
            print(f"[ERROR] Processing item {idx}: {e}")

    if not debug:
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"\n--- Benchmark Results ---")
        print(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA visual analogy benchmark (direct prompt + debug mode).")
    parser.add_argument('--benchmark_file', type=str, required=True, help='Path to the benchmark JSON file')
    parser.add_argument('--visual_debug', action='store_true', help='Use detailed visual debug prompt for model pipeline testing')
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

    run_kiva_benchmark(data, model, processor, debug=args.visual_debug)

if __name__ == "__main__":
    main()
