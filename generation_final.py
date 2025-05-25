import requests
import random
import multiprocessing
from story_config import NAME_LIST
# from story_doubles import sentence_question_pairs
from story_doubles_new import sentence_question_pairs

def _generate_story_process(result_queue):
    try:
        print("Making a story")
        # Select random names and sentence-question pair
        name1, name2 = random.sample(NAME_LIST, 2)
        pair_id = random.choice(list(sentence_question_pairs.keys()))
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
        
        # API call
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
                # "frequency_penalty": 0.5,
                # "presence_penalty": 0.2
            },
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            },
            timeout=4  # API call timeout
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        
        story_text = response.json()['choices'][0]['message']['content'].strip()
        
        result = {
            'story': story_text,
            'names': [name1, name2],  # Our pre-selected names
            'question': formatted_question,  # Pre-formatted question
            'original_sentence': formatted_sentence
        }
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def generate_story():
    while True:
        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_generate_story_process, args=(result_queue,))
        p.start()
        # Wait up to 30 seconds for the process to complete
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join()
            print("Process took too long. Resetting generation process.")
            continue  # Restart the process immediately
        result = result_queue.get()
        if isinstance(result, Exception):
            print(f"Error occurred: {result}. Restarting generation process.")
            continue
        return result

if __name__ == "__main__":
    result = generate_story()
    print("Generated Story:")
    print(result['story'])
    print("\nNames:", result['names'])
    print("Question:", result['question'])
    print("Original Sentence:", result['original_sentence'])
