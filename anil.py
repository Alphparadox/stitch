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
            question = item["question"]
            gt = item["ground_truth_answer"].strip().upper()

            # ==========================================================
            # ⚡ Optimized concise prompt for higher accuracy & speed
            # ==========================================================
            prompt_content = f"""<image>\n{question}
Answer only with the correct option: A, B, or C."""

            # ==========================================================
            inputs = processor(
                text=prompt_content,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=8)

            ans_raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            if "ASSISTANT:" in ans_raw:
                ans_raw = ans_raw.split("ASSISTANT:")[-1].strip()

            print(f"Question: {question}")
            print(f"Ground Truth: {gt}")
            print(f"Model Answer (Raw): {ans_raw}")

            choice = next((c for c in ans_raw.upper() if c in ("A", "B", "C")), "")
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
