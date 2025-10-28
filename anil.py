import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse
import re

# =============================================================================
# LLaVA Visual Analogy Benchmark Runner (Rotation-Focused CoT Prompt Version)
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

def build_cot_prompt_with_rotation_focus():
    return (
        "<image>\n"
        "You are shown a visual analogy problem with two rows.\n\n"
        "1. Carefully analyze the top row:\n"
        "- Describe the left image in detail.\n"
        "- Describe the right image in detail.\n"
        "- Determine if any object was rotated, reflected, or transformed.\n"
        "- If a rotation is present, estimate the angle (in degrees) and the direction (clockwise or counter-clockwise).\n"
        "- Clearly state the transformation rule you observe (e.g., 'The top images differ by a 90-degree clockwise rotation').\n\n"
        "2. Now, examine the bottom row:\n"
        "- Focus on the left image. Imagine applying the same type of transformation (with the same degree and direction) to it.\n"
        "- Describe your predicted result after applying the transformation.\n\n"
        "3. Among the options (A, B, or C), which image shows the correct outcome after this specific transformation? Explain why the selected option matches the rule.\n\n"
        "Finish your answer with: \"The correct option is [A/B/C].\""
    )

def extract_answer(raw_text):
    pattern = r"The correct option is\s*\[*([ABC])\]*"
    matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Fallback: use last mention of A/B/C
    for char in reversed(raw_text.upper()):
        if char in "ABC":
            return char
    return ""

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

            prompt = build_cot_prompt_with_rotation_focus()
            gt = sample["ground_truth_answer"].strip().upper()

            inputs = processor(
                text=prompt,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=150)
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
    parser = argparse.ArgumentParser(description="Run LLaVA visual analogy benchmark (rotation-aware prompt).")
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
