import json
import os

def extract_names_from_json(json_folder):
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            print(f"\nFile: {filename}")
            with open(os.path.join(json_folder, filename)) as f:
                data = json.load(f)
                
            characters = data.get('characters', [])
            names = []
            
            for char in characters:
                name = char.get('character', 'Unknown')
                names.append(name)
            
            print("Characters found:", ", ".join(names))

if __name__ == "__main__":
    extract_names_from_json("/root/llama/llm_results")