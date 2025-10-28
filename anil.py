import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse
import re

# =============================================================================
# Improved LLaVA Benchmark Runner (CoT, Modular Prompt, Robust Parsing)
# =============================================================================

BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_DIR = "/home/naveenkumar/stitch"

def get_full_image_path(relative_path):
    path = os.path.join(BASE_IMAGE_DIR, relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    return path

def load_benchmark_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None

def build_cot_prompt():
    # Modular reasoning structure for analogies
    return (
        "<image>\n"
        "You are shown a visual analogy puzzle problem.\n"
        "<SUMMARY> Describe how the top left image transforms into the top right image. What rule is applied? </SUMMARY>\n"
        "<CAPTION> List and describe all objects, orientations, and visual features in every panel. </CAPTION>\n"
        "<REASONING> Apply the top row's transformation to the lower left image step-by-step. What should change? </REASONING>\n"
        "<CONCLUSION> Select the correct option (A, B, or C) matching the transformation and explain your choice. End with: The correct option is [A/B/C]. </CONCLUSION>\n"
    )

def extract_answer(raw_text):
    # First, try strict regex on the expected conclusion phrase
    pattern = r"The correct option is\s*\[*([ABC])\]*"
    matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Fallback: use last mention of A/B/C
    candidate = ""
    for char in reversed(raw_text.upper()):
        if char in "ABC":
            candidate = char
            break
    return candidate

def run_kiva_benchmark(benchmark_data, model, processor):
    print(f"\n--- LLaVA Visual Analogy Benchmark ---")
    correct = 0
    total = len(benchmark_data)

    for idx, sample in enumerate(benchmark_data, 1):
        print(f"\n[Item {idx}/{total}]")
        try:
            img_path = get_full_image_path(sample["image"])
            img = load_benchmark_image(img_path)
            if img is None:
                continue

            prompt = build_cot_prompt()
            gt = sample["ground_truth_answer"].strip().upper()

            # Prepare input batch
            inputs = processor(
                text=prompt,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=120)
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
    parser = argparse.ArgumentParser(description="Run LLaVA visual analogy benchmark.")
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
