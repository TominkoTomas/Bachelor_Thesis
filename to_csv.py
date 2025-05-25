import json
import os

def combine_json_files(input_folder, output_file):
    combined_data = []
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            
            # Load JSON data from the file
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.append(data)
    
    # Write the combined data to the output file
    with open(output_file, 'w') as f_out:
        json.dump(combined_data, f_out, indent=2)

# Example usage
input_folder = 'llm_results_old_model_final'
output_file = 'combined_output.json'
combine_json_files(input_folder, output_file)
