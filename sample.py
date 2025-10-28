import os
import json
import argparse
import re

# =============================================================================
#
# BENCHMARK JSON GENERATOR
#
# This script:
# 1. Scans an --input_directory for image files (e.g., "2DRotation+90_19_0_(A).jpg").
# 2. Parses the filename to extract:
#    - The transformation (e.g., "2DRotation+90")
#    - The ground truth answer (e.g., "A")
# 3. Generates a question from the transformation.
# 4. Creates a 'benchmark_data.json' file that the LLaVA script can use.
#
# =============================================================================

def generate_question_from_filename(transformation_str):
    """Creates a simple question based on the transformation string."""
    # Example: "2DRotation+90" becomes 
    # "Which option (A, B, or C) shows the 2DRotation+90 transformation?"
    return f"Which option (A, B, or C) shows the {transformation_str} transformation?"

def generate_benchmark_json(input_directory, output_directory):
    """
    Scans the input directory, parses filenames, and generates the benchmark.json.
    """
    
    # Regex to parse filenames like: 2DRotation+90_19_0_(A).jpg
    # Group 1: Transformation (2DRotation+90)
    # Group 2: Index 1 (19)
    # Group 3: Index 2 (0)
    # Group 4: Ground Truth (A)
    # Allows for .jpg, .jpeg, or .png extensions (case-insensitive)
    filename_regex = re.compile(r"^(.*?)_([0-9]+)_([0-9]+)_\(([A-Z])\)\.(jpg|jpeg|png)$", re.IGNORECASE)

    benchmark_data = []
    
    # Get the name of the input directory itself (e.g., "harsh")
    # This is used to create the relative path for the JSON file.
    input_dir_name = os.path.basename(os.path.normpath(input_directory))

    print(f"Scanning directory: {input_directory}")

    for filename in os.listdir(input_directory):
        match = filename_regex.match(filename)
        
        if match:
            # Extract data from the filename
            transformation = match.group(1)
            ground_truth = match.group(4)
            
            # Generate the question
            question = generate_question_from_filename(transformation)
            
            # Create the relative image path for the JSON
            # e.g., "harsh/2DRotation+90_19_0_(A).jpg"
            relative_image_path = os.path.join(input_dir_name, filename)
            
            # Add this item to our benchmark list
            benchmark_data.append({
                "image": relative_image_path,
                "question": question,
                "ground_truth_answer": ground_truth
            })
            
        else:
            print(f"Skipping file (does not match expected format): {filename}")

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Write the final JSON file
    json_output_path = os.path.join(output_directory, "benchmark_data.json")
    
    with open(json_output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\nSuccessfully generated {json_output_path}")
    print(f"Total benchmark items created: {len(benchmark_data)}")

# -------------------------------
# Main execution block
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark JSON from a directory of pre-named images.")
    parser.add_argument('--input_directory', type=str, required=True,
                        help='Path to the directory containing your pre-generated composite images.')
    parser.add.argument('--output_directory', type=str, required=True,
                        help='Path to save the generated benchmark_data.json file.')
    args = parser.parse_args()

    generate_benchmark_json(args.input_directory, args.output_directory)

