import requests
import random
import multiprocessing
from story_config import NAME_LIST
from story_doubles import sentence_question_pairs

def _generate_story_process(result_queue, pair_id): 
    try:
        print(f"Making a story for pair {pair_id}")
        # Select random names
        name1, name2 = random.sample(NAME_LIST, 2)
        # Use the provided pair_id
        pair = sentence_question_pairs[pair_id]
        
        # Format the sentence and question with names
        formatted_sentence = pair["sentence"].format(name1=name1)
        formatted_question = pair["question"]  # Using name1 as default for {name}

        prompt = f"""Write a 200-word fictional basic story using these characters: {name1} and {name2}.

        STRICT REQUIREMENTS:
        1. MUST include this exact sentence: "{formatted_sentence}"
        2. Style: Simple language (grade school level)
        3. Keep names consistent: {name1} and {name2}
        4. Make the story logically include the sentence
        5. Dont give the {name2} any specific trait or action related to the exact sentence mentioned or its context
        6. Output the story as whole dont use newliners (\n).
        7. LANGUAGE: ENGLISH ONLY, NO Chinese characters

        JUST OUTPUT THE STORY TEXT, nothing else. Do not include any metadata or formatting."""
        
        # API call (unchanged)
        API_KEY = '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'
        API_URL = 'https://openrouter.ai/api/v1/chat/completions'
        response = requests.post(
            API_URL,
            json={
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [
                    {"role": "system", "content": "You are a story writer who only uses simple English."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.2,
                "top_p": 0.95,
                "max_tokens": 700,
            },
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            timeout=4
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        story_text = response.json()['choices'][0]['message']['content'].strip()
        
        result = {
            'story': story_text,
            'names': [name1, name2],
            'question': formatted_question,
            'original_sentence': formatted_sentence,
            'pair_id': pair_id  
        }
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def generate_story(pair_id):  
    while True:
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_generate_story_process, args=(result_queue, pair_id))
        p.start()
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join()
            print("Process took too long. Resetting generation process.")
            continue
        result = result_queue.get()
        if isinstance(result, Exception):
            print(f"Error occurred: {result}. Restarting generation process.")
            continue
        return result

if __name__ == "__main__":
    # Example usage for testing
    ordered_pair_ids = list(sentence_question_pairs.keys())
    result = generate_story(ordered_pair_ids[0])
    print(result)
    print("Generated Story:")
    print(result['story'])
    print("\nNames:", result['names'])
    print("Question:", result['question'])
    print("Original Sentence:", result['original_sentence'])