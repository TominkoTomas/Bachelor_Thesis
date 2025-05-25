from llama_cpp import Llama
import matplotlib.pyplot as plt
import numpy as np
import generation_all
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

def analyze_response(llm, prompt, run_visualization=False):
    """Run analysis for a given prompt and return structured data"""
    output = llm(
        prompt,
        max_tokens=500,
        stop=["Q:", "\n"],
        echo=True,
        logprobs=True,  # Only get the best logprob
        temperature=2.2,
        top_p = 1,
        # frequency_penalty=0.5,  # Encourage repetition
        # presence_penalty=0.5,   # Encourage common phrases
    )

    # Extract data from response
    generated_text = output['choices'][0]['text']
    all_tokens = output['choices'][0]['logprobs']['tokens']
    all_logprobs = output['choices'][0]['logprobs']['token_logprobs']
    
    # Find where the answer starts in the token stream
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    num_prompt_tokens = len(prompt_tokens)
    
    # Extract just the answer tokens
    answer_tokens = all_tokens[num_prompt_tokens -1 :]
    answer_logprobs = all_logprobs[num_prompt_tokens - 1:]
    
    # Prepare simplified token data 
    token_data = []
    for token, logprob in zip(answer_tokens, answer_logprobs):
        # Convert any numpy types to Python native types
        log_prob_value = float(logprob) if logprob is not None else None
        token_data.append({
            "token": token,
            "logprob": log_prob_value
        })

    # Only run visualization if requested
    if run_visualization:
        visualize_response(llm, prompt, all_tokens, all_logprobs, output['choices'][0]['logprobs']['top_logprobs'], generated_text)
    
    # Extract the actual answer from the generated text
    answer = extract_answer(generated_text)
    
    return {
        "generated_text": generated_text,
        "answer": answer,
        "token_data": token_data
    }

def extract_answer(text):
    """Extract just the answer part from the generated text"""
    try:
        # Find the part after "A:" in the text
        answer_part = text.split("A:", 1)[1].strip()
            
        return answer_part
    except:
        # Return the whole text if we can't parse it properly
        return text

def visualize_response(llm, prompt, all_tokens, all_logprobs, all_top_logprobs, generated_text):
    """Visualize the response data"""
    # Visualization logic
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    num_prompt_tokens = len(prompt_tokens)
    gen_tokens = all_tokens[num_prompt_tokens -1:]
    gen_logprobs = all_logprobs[num_prompt_tokens -1:]
    gen_top_logprobs = all_top_logprobs[num_prompt_tokens -1:]

    valid_indices = [i for i, x in enumerate(gen_logprobs) if x is not None]
    gen_tokens = [gen_tokens[i] for i in valid_indices]
    gen_logprobs = [gen_logprobs[i] for i in valid_indices]
    gen_top_logprobs = [gen_top_logprobs[i] for i in valid_indices]

    best_logprobs = [max(t.values()) if t else None for t in gen_top_logprobs]
    
    # Plotting
    plt.figure(figsize=(20, 10))
    x_values = np.arange(len(gen_tokens))
    
    plt.plot(x_values, gen_logprobs, marker='o', linestyle='-', color='blue', label='Actual Token LogProb')
    plt.scatter(x_values, best_logprobs, color='red', s=40, marker='x', label='Best Possible LogProb')
    
    plt.xticks(x_values, [t.replace(' ', '" "') for t in gen_tokens], rotation=90, fontsize=10)
    plt.gca().xaxis.grid(True, linestyle='--', alpha=0.8, linewidth=1.5)
    plt.gca().yaxis.grid(True, linestyle='-', alpha=0.6, linewidth=0.8)
    
    plt.xlabel("Tokens", fontsize=12)
    plt.ylabel("Log Probability", fontsize=12)
    plt.title("Token Probability Analysis: Actual vs Best", fontsize=14)
    plt.ylim(min(gen_logprobs + best_logprobs) - 0.15, max(gen_logprobs + best_logprobs) + 0.3)
    plt.legend()
    
    print("\nGenerated Text:", generated_text)
    plt.tight_layout()
    plt.show()

def run_multiple_times(llm, prompt, question, name, story, num_runs=5, run_visualization=False):
    """Run the same question multiple times and collect results"""
    answers = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs} for question: {question} (Character: {name})")
        
            
        result = analyze_response(llm, prompt, run_visualization=run_visualization)
        
        # Create a simpler answer structure with only the requested fields
        answer_data = {
            "run_number": run + 1,
            "answer": result["answer"],
            "token_data": result["token_data"]
        }
        
        answers.append(answer_data)
        
        # Print a quick summary
        print(f"Answer for {name}: {result['answer']}")
        
    # Return a question-focused structure
    return {
        "question": question,
        "character": name,
        "story": story,
        "answers": answers
    }

def save_results_to_json(results, filename=None):
    """Save results to a JSON file"""
    if filename is None:
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_results_final_{timestamp}.json"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Use the custom encoder to handle numpy types
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

def main(num_stories=10, runs_per_question=5, enable_visualization=False, n_ctx=4096):
    # Initialize LLM
    llm = Llama.from_pretrained(
        repo_id="QuantFactory/Llama-3.2-3B-GGUF",
        filename="Llama-3.2-3B.Q2_K.gguf",
        n_gpu_layers=-1,
        n_ctx=n_ctx,  # Set context window size
        verbose=False,  # Set to True if you want verbose output
        logits_all=True,
        temperature = 3,
        top_p = 1,
        presence_penalty = 0.5,
        frequency_penalty = 0.5,
    )

    results = []
    story_count = 78 # change to 0
    total_generations = 0
    
    # Create a results directory
    results_dir = "llm_results_old_model_final"
    os.makedirs(results_dir, exist_ok=True)

    ordered_pair_ids = list(generation_all.sentence_question_pairs.keys())
    
    while story_count < num_stories :
        try:
            # Get the current pair_id 
            current_pair_id = ordered_pair_ids[story_count % len(ordered_pair_ids)]

            # Generate story data
            story_data = generation_all.generate_story(current_pair_id)
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
                prompt = f"""Story Context:\n{story_data['story']}\n\n 
                        - IMPORTANT: Answer using one whole sentance. 
                        \n\n Q: {question} A:"""
                
                print(f"\n{'='*80}")
                print(f"Analyzing: {question}")
                
                # Only run visualization if enabled and this is the first run
                run_viz = enable_visualization and (story_count == 0 and name == story_data['names'][0])
                
                # Run the question multiple times
                character_result = run_multiple_times(
                    llm, 
                    prompt, 
                    question, 
                    name,
                    story_data['story'],
                    num_runs=runs_per_question,
                    run_visualization=run_viz
                )
                
                # Add to the story results
                story_results["characters"].append(character_result)
                
                total_generations += runs_per_question
                

            
            # Add the completed story results to the overall results
            results.append(story_results)
            
            # Save intermediate results after each story
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_filename = os.path.join(results_dir, f"story_{story_count+1}_{timestamp}.json")
            save_results_to_json(story_results, intermediate_filename)
            
            story_count += 1
        
        except Exception as e:
            # Failed to generate story continue with trying to generate replacement more.
            print(f"Retrying story {story_count+1}...")
            continue
    
    # Save all results to a final JSON file
    final_filename = os.path.join(results_dir, f"llm_results_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results_to_json(results, final_filename)
    
    print(f"\nCompleted {total_generations} total generations across {story_count} stories")
    print(f"Final results saved to {final_filename}")

if __name__ == "__main__":
    # Parameters to adjust
    main(num_stories=100, runs_per_question=10, enable_visualization=False, n_ctx=8000)