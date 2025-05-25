import json
import csv

# Load JSON data
with open('llm_results_old_model_final/combined_output.json', 'r') as f:
    data = json.load(f)

# --- File 1: stories.csv ---
with open('data/stories_big.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['story_id', 'story'])
    for story_id, story in enumerate(data, start=1):
        writer.writerow([story_id, story['story']])  # story_id maps to full text

# --- File 2: answers.csv ---
with open('data/answers_big.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['story_id', 'character', 'question', 'run_number', 'answer', 'token_data', 'is_hallucinated', 'use_for_training'])
    for story_id, story in enumerate(data, start=1):
        for character_idx, character in enumerate(story['characters'], start=1):
            for answer in character['answers']:
                # Determine if hallucinated based on character position (odd/even)
                is_hallucinated = 0 if character_idx % 2 == 1 else 1
                writer.writerow([
                    story_id,
                    character['character'],
                    character['question'],
                    answer['run_number'],
                    answer['answer'],
                    json.dumps(answer['token_data']),  # Serialize token_data
                    is_hallucinated,  # 0 for odd positions, 1 for even
                    0   # Default: 1 (use for training)
                ])