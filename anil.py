import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse

# =============================================================================
# LLaVA SINGLE-IMAGE BENCHMARK RUNNER SCRIPT
# =============================================================================

BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_IMAGE_DIR = "/home/naveenkumar/fyp"


def get_full_image_path(relative_path):
    """Construct the absolute path for an image using the base directory."""
    return os.path.join(BASE_IMAGE_DIR, relative_path)


def load_benchmark_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
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

        image_path = get_full_image_path(item["image"])
        img = load_benchmark_image(image_path)

        if not img:
            print(f"Skipping item {idx} due to missing image.")
            continue

        try:
            question = item["question"]
            ground_truth = item["ground_truth_answer"].strip().upper()

            prompt_content = f"""USER: <image>\n{question}
Please answer with only the letter 'A', 'B', or 'C'.
ASSISTANT:"""

            inputs = processor(
                text=prompt_content,
                images=[img],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)

            model_answer_raw = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            if "ASSISTANT:" in model_answer_raw:
                model_answer_raw = model_answer_raw.split("ASSISTANT:")[-1].strip()

            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Answer (Raw): {model_answer_raw}")

            model_choice = ""
            for char in model_answer_raw.upper():
                if char in ('A', 'B', 'C'):
                    model_choice = char
                    break

            if model_choice == ground_truth:
                print(f"Result: CORRECT ✅ (Model chose {model_choice})")
                correct += 1
            else:
                print(f"Result: INCORRECT ❌ (Expected: {ground_truth}, Got: {model_choice or 'None'})")

        except Exception as e:
            print(f"Error processing item {idx}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n--- Benchmark Complete ---")
    print(f"Total Items: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA single-image benchmark from a JSON file.")
    parser.add_argument('--benchmark_file', type=str, required=True,
                        help='Path to the benchmark_data.json file.')
    args = parser.parse_args()

    print("--- PART 1: LOADING BENCHMARK DATA ---")
    try:
        with open(args.benchmark_file, 'r') as f:
            benchmark_data_list = json.load(f)
        print(f"Loaded {len(benchmark_data_list)} items from {args.benchmark_file}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load benchmark file from {args.benchmark_file}.")
        print(f"Error: {e}")
        return

    print("\n--- PART 2: LOADING LLaVA MODEL ---")
    print(f"Loading base LLaVA model from {BASE_MODEL_PATH}...")

    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=dtype
        ).to(DEVICE)

        print(f"Model loaded to {DEVICE} ✅")

    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model from {BASE_MODEL_PATH}.")
        print(f"Error: {e}")
        return

    # --- PART 3: Run Benchmark ---
    run_kiva_benchmark(benchmark_data_list, model, processor)


if __name__ == "__main__":
    main()
