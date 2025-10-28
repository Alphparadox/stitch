import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse

# =============================================================================
# LLaVA SINGLE-IMAGE BENCHMARK RUNNER SCRIPT (Fast & Accurate Version)
# =============================================================================

BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_DIR = "/home/naveenkumar/stitch"


def get_full_image_path(relative_path):
    return os.path.join(BASE_IMAGE_DIR, relative_path)


def load_benchmark_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def run_kiva_benchmark(benchmark_data, model, processor):
    print(f"\n--- Running Single-Image Benchmark on model: {BASE_MODEL_PATH} ---")
    correct = 0
    total = len(benchmark_data)

    if total == 0:
        print("Benchmark dataset is empty.")
        return

    for idx, item in enumerate(benchmark_data, 1):
        print(f"\n--- Test Item {idx}/{total} ---")

        img_path = get_full_image_path(item["image"])
        img = load_benchmark_image(img_path)
        if not img:
            print(f"Skipping item {idx} due to missing image.")
            continue

        try:
            # We don't use the item["question"] because our new prompt is more robust
            gt = item["ground_truth_answer"].strip().upper()

            # ==========================================================
            # ⚡ NEW/MODIFIED: Chain of Thought (CoT) Prompt
            # This forces the model to analyze the rule and then apply it.
            # ==========================================================
            prompt_content = f"""<image>
The image shows a visual analogy problem.
1. First, analyze the top example to find the transformation rule.
2. Second, apply that exact rule to the left image in the bottom options.
3. Finally, select the option (A, B, or C) that correctly shows the result.

Explain your reasoning step-by-step and conclude with "The correct option is [letter]".
"""
            
            # ==========================================================
            inputs = processor(
                text=prompt_content,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():
                # ==========================================================
                # ⚡ NEW/MODIFIED: Increased max_new_tokens for reasoning
                # ==========================================================
                output_ids = model.generate(**inputs, max_new_tokens=150)

            ans_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            if "ASSISTANT:" in ans_raw:
                ans_raw = ans_raw.split("ASSISTANT:")[-1].strip()

            print(f"Ground Truth: {gt}")
            print(f"Model Answer (Raw): {ans_raw}")
            
            # ==========================================================
            # ⚡ NEW/MODIFIED: Robust CoT Parsing
            # We find the *last* mention of A, B, or C, as that
            # will be in the conclusion after all the reasoning.
            # ==========================================================
            choice = ""
            for char in reversed(ans_raw.upper()):
                if char in ("A", "B", "C"):
                    choice = char
                    break

            if choice == gt:
                print(f"Result: CORRECT ✅ ({choice})")
                correct += 1
            else:
                print(f"Result: INCORRECT ❌ (Expected: {gt}, Got: {choice or 'None'})")

        except Exception as e:
            print(f"Error processing item {idx}: {e}")

    acc = (correct / total) * 100 if total > 0 else 0
    print("\n--- Benchmark Complete ---")
    print(f"Total Items: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA single-image benchmark from a JSON file.")
    parser.add_argument('--benchmark_file', type=str, required=True, help='Path to benchmark_data.json')
    args = parser.parse_args()

    print("--- Loading Benchmark Data ---")
    try:
        with open(args.benchmark_file, 'r') as f:
            benchmark_data_list = json.load(f)
        print(f"Loaded {len(benchmark_data_list)} items.")
    except Exception as e:
        print(f"Error loading benchmark file: {e}")
        return

    print("\n--- Loading LLaVA Model ---")
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, torch_dtype=dtype).to(DEVICE)
        print(f"Model loaded on {DEVICE} ✅")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    run_kiva_benchmark(benchmark_data_list, model, processor)


if __name__ == "__main__":
    main()