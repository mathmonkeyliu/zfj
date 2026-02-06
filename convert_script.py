import json
import os

def convert_file(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return

    print(f"Converting {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    new_data = {}
    for k, v in data.items():
        # Check format
        if "," in k:
            new_key = k.replace(",", "")
            new_data[new_key] = v
        else:
            new_data[k] = v
            
    # Backup original
    os.rename(filename, filename + ".bak")
    
    with open(filename, 'w') as f:
        json.dump(new_data, f)
    print(f"Done. Original backed up to {filename}.bak")

if __name__ == "__main__":
    convert_file("topk_1.json")
    convert_file("topk_2.json")
