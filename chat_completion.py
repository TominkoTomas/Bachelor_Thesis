from llama_cpp import Llama
import matplotlib.pyplot as plt
import numpy as np
import generation_final
import json
import os
from datetime import datetime
import time

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def create_chat_prompt(story, question):
    """Create properly formatted chat messages"""
    # "You are a question answering helper. Analyze the story and answer questions given to you. If there is no answer to the question anser something anyways. Always answer using one short sentance."
    return [
        {"role": "system", "content": "Answer all questions using the story context. If exact information is missing, make reasonable assumptions based on context. Never mention information gaps - always provide a concise answer."},
        {"role": "user", "content": f"Story Context:\n{story}\n\nQuestion: {question}"}
    ]

def analyze_response(llm, messages, run_visualization=False):
    """Run analysis using chat completion"""
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=500,
        temperature = 2.2,
        top_p = 1,
        frequency_penalty=1,  # Encourage repetition
        presence_penalty=1,   # Encourage common phrases
        stop=["<|eot_id|>"],  # Model-specific stop token
        logprobs=True,  # Enable logprobs
        top_logprobs=0,  # Number of top logprobs to return per token
        logit_bias={"128000": -100.0},  # Prevent <|end_of_text|> generation
    )

    # Extract data from response format
    choice = output['choices'][0]
    generated_text = choice['message']['content']
    logprobs = choice.get('logprobs', {}).get('content', [])
    
    # Process token data
    token_data = []
    for token_info in logprobs:
        if not token_info:
            continue
            
        # Get main token info
        token_data.append({
            "token": token_info.get('token', ''),
            "logprob": float(token_info['logprob']) if 'logprob' in token_info else None
        })
        

    # Visualization logic
    if run_visualization:
        visualize_response(llm, messages, generated_text, token_data)
        
    return {
        "generated_text": generated_text,
        "answer": generated_text,
        "token_data": token_data
    }

def visualize_response(llm, messages, generated_text, token_data):
    """Visualize the response data for chat completion"""
    full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt_tokens = llm.tokenize(full_prompt.encode("utf-8"))
    
    # Extract just the answer tokens
    answer_tokens = [t["token"] for t in token_data]
    answer_logprobs = [t["logprob"] for t in token_data]
    
    # Plotting
    plt.figure(figsize=(20, 10))
    x_values = np.arange(len(answer_tokens))
    
    plt.plot(x_values, answer_logprobs, marker='o', linestyle='-', color='blue', label='Token LogProb')
    
    plt.xticks(x_values, [t.replace(' ', '" "') for t in answer_tokens], rotation=90, fontsize=10)
    plt.gca().xaxis.grid(True, linestyle='--', alpha=0.8, linewidth=1.5)
    plt.gca().yaxis.grid(True, linestyle='-', alpha=0.6, linewidth=0.8)
    
    plt.xlabel("Tokens", fontsize=12)
    plt.ylabel("Log Probability", fontsize=12)
    plt.title("Token Probability Analysis", fontsize=14)
    plt.ylim(min(answer_logprobs) - 0.15, max(answer_logprobs) + 0.3)
    plt.legend()
    
    print("\nGenerated Text:", generated_text)
    plt.tight_layout()
    plt.show()

def run_multiple_times(llm, messages, question, name, story, num_runs=5, run_visualization=False):
    """Run the same question multiple times and collect results"""
    answers = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs} for question: {question} (Character: {name})")
            
        result = analyze_response(llm, messages, run_visualization=run_visualization)
        
        # Create answer structure
        answer_data = {
            "run_number": run + 1,
            "answer": result["answer"],
            "token_data": result["token_data"]
        }
        
        answers.append(answer_data)
        print(f"Answer for {name}: {result['answer']}")
        
    return {
        "question": question,
        "character": name,
        "story": story,
        "answers": answers
    }

def save_results_to_json(results, filename=None):
    """Save results to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_results_final_{timestamp}.json"
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

def main(num_stories=10, runs_per_question=5, enable_visualization=False, n_ctx=4096):
    # Initialize LLM with chat format
    llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        presence_penalty = 0.7,
        frequency_penalt = 0.7,
        chat_format="llama-3",
        verbose=False,
        logits_all=True,
        temeprature = 3,
        top_p = 1
    )

    results = []
    story_count = 0
    total_generations = 0
    results_dir = "llm_results_final"
    os.makedirs(results_dir, exist_ok=True)
    while story_count < num_stories:
        try:
            # Generate story data
        
            story_data = generation_final.generate_story()
            print(f"\n{'='*80}")
            print(f"Story {story_count+1}/{num_stories}:")
            print(story_data['story'])
            
            story_results = {
                "story": story_data['story'],
                "characters": []
            }
            
            # Process both characters
            for name in story_data['names']:
                question = story_data['question'].format(name=name)
                messages = create_chat_prompt(story_data['story'], question)
                
                print(f"\n{'='*80}")
                print(f"Analyzing: {question}")
                
                # Visualization control
                run_viz = enable_visualization and (story_count == 0 and name == story_data['names'][0])
                
                # Run multiple times
                character_result = run_multiple_times(
                    llm, 
                    messages,
                    question, 
                    name,
                    story_data['story'],
                    num_runs=runs_per_question,
                    run_visualization=run_viz
                )
                
                story_results["characters"].append(character_result)
                total_generations += runs_per_question

            # Save progress
            results.append(story_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_filename = os.path.join(results_dir, f"story_{story_count+1}_{timestamp}.json")
            save_results_to_json(story_results, intermediate_filename)
            
            story_count += 1
        
        except Exception as e:
            print(f"Retrying story {story_count+1}... Error: {str(e)}")
            continue

    # Final save
    final_filename = os.path.join(results_dir, f"llm_results_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results_to_json(results, final_filename)
    print(f"\nCompleted {total_generations} generations across {story_count} stories")

if __name__ == "__main__":
    main(
        num_stories=1,
        runs_per_question=5,
        enable_visualization=False,
        n_ctx=8000
    )