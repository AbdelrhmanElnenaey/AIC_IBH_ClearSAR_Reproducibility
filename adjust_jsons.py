import json
import os
import glob

# Replace with the path to your folder containing the JSON files
folder_path = "mmdetection-output/"

def modify_predictions(filepath):
    print(f"\n--- Processing: {filepath} ---")
    try:
        # 1. Read the JSON file into memory
        with open(filepath, 'r') as f:
            predictions = json.load(f)
            
        print(f"Successfully loaded {len(predictions)} predictions.")
        print("Subtracting 50000 from image_ids...")
        
        # 2. Iterate and modify the data in-place
        for pred in predictions:
            # Subtract 50000 from the image_id
            if 'image_id' in pred:
                pred['image_id'] = pred['image_id'] - 50000

        # 3. Overwrite the original file with the updated data
        print(f"Overwriting {filepath}...")
        with open(filepath, 'w') as f:
            # Omitting 'indent=4' here to keep the file size as small as possible
            json.dump(predictions, f)
            
        print("Done! File updated.")
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

def process_folder(folder):
    print(f"Scanning directory: {folder}")
    
    # Create a search pattern to find all .json files in the folder
    search_pattern = os.path.join(folder, '*.json')
    json_files = glob.glob(search_pattern)
    
    # Check if we actually found anything
    if not json_files:
        print(f"⚠️ No JSON files found in directory: {folder}")
        return

    print(f"Found {len(json_files)} JSON file(s). Starting batch modification...")
    
    # Loop through each file and apply the modification
    for file_path in json_files:
        modify_predictions(file_path)
        
    print("\n✅ All files in the folder have been processed!")

if __name__ == "__main__":
    process_folder(folder_path)