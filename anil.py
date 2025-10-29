import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse
import re

# ==============================================================
# CONFIG
# ==============================================================

BASE_MODEL_PATH = "llava-hf/llava-v1.6-vicuna-13b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_DIR = "/home/naveenkumar/stitch"
MODEL_INPUT_SIZE = 336  # fixed for llava-1.6

# ==============================================================

def get_full_image_path(relative_path):
    path = os.path.join(BASE_IMAGE_DIR, relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return path

def load_benchmark_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"[Image] Path: {image_path} | Size: {img.size}")
        if img.size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
            img.thumbnail((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
            bg = Image.new("RGB", (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (255, 255, 255))
            bg.paste(img, ((MODEL_INPUT_SIZE - img.size[0]) // 2, (MODEL_INPUT_SIZE - img.size[1]) // 2))
            img = bg
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return None

def build_prompt(question):
    return (
        f"<image>\n"
        f"{question}\n\n"
        "Choose the correct answer among (A), (B), and (C).\n"
        "Respond with only one capital letter — A, B, or C. Do not explain."
    )

def extract_answer(raw_text):
    match = re.search(r"\b([ABC])\b", raw_text.strip().upper())
    return match.group(1) if match else ""

def run_benchmark(benchmark_data, model, processor):
    print("\n--- LLaVA Visual Analogy Benchmark ---")
    correct, total = 0, len(benchmark_data)

    for idx, sample in enumerate(benchmark_data, 1):
        print(f"\n[Item {idx}/{total}]")
        try:
            img_path = get_full_image_path(sample["image"])
            img = load_benchmark_image(img_path)
            if img is None:
                continue

            question = sample["question"]
            gt = sample["ground_truth_answer"].strip().upper()

            prompt = build_prompt(question)

            # --- THIS IS THE CORRECTED LINE ---
            inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
            # ------------------------------------
            
            print("[Batch] pixel_values:", inputs["pixel_values"].shape)

            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=50)
            raw_answer = processor.decode(output[0], skip_special_tokens=True).strip()

            choice = extract_answer(raw_answer)
            print(f"GT: {gt} | Model: {raw_answer} | Extracted: {choice}")

            if choice == gt:
                print("✅ CORRECT")
                correct += 1
            else:
                print(f"❌ INCORRECT — expected {gt}")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Processing item {idx}: {e}")

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\n--- Benchmark Results ---\nTotal: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_file", type=str, required=True)
    args = parser.parse_args()

    print("Loading benchmark data...")
    with open(args.benchmark_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    run_benchmark(data, model, processor)

if __name__ == "__main__":
    main()