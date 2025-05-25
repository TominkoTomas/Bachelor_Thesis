import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch
from names_logprobs import NAME1, NAME2, LOGPROBS1, LOGPROBS2

length_of_all = len(LOGPROBS1)

name1 = [n for n in NAME1 for _ in range(length_of_all)]  # List of names repeated 5 times
name2 = [n for n in NAME2 for _ in range(length_of_all)]  # List of names repeated 5 times
logprobs1 = LOGPROBS1  # Logprobs for hallucinated names
logprobs2 = LOGPROBS2  # Logprobs for valid names

def generate_comparisons():
    os.makedirs('bar_plots', exist_ok=True)
    os.makedirs('histograms', exist_ok=True)
    
    # Collect all logprobs for histograms per story
    hist_data = defaultdict(lambda: {'logprobs1': [], 'logprobs2': []})
    
    for idx in range(length_of_all):
        story_num = (idx // 5) + 1  # Calculate story number
        run_num = (idx % 5) + 1     # Calculate run number within story
        
        # Extract the token-logprob pair for this run
        token1, logprob1 = LOGPROBS1[idx]
        token2, logprob2 = LOGPROBS2[idx]
        
        # Get the names for this story
        name_idx = idx // 5  # Get the index for names 
        hallucinated_name = NAME1[name_idx]
        valid_name = NAME2[name_idx]
        
        # ========== BAR PLOT (2 bars per run) ==========
        plt.figure(figsize=(10, 5))
        tokens = [f"{token1}\n({hallucinated_name})", f"{token2}\n({valid_name})"]
        logprobs = [logprob1, logprob2]
        colors = ['red', 'blue']
        
        bars = plt.bar(range(2), logprobs, color=colors, alpha=0.7)
        plt.xticks(range(2), tokens, rotation=45, ha='right')
        
        # Add legend and labels
        plt.title(f'Log Probability Comparison - Story {story_num} Run {run_num}')
        plt.ylabel('Log Probability')
        plt.xlabel('Token (Character Name)')
        
        # Create custom legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label=f'Hallucinated ({hallucinated_name})'),
            Patch(facecolor='blue', alpha=0.7, label=f'Valid ({valid_name})')
        ]
        plt.legend(handles=legend_elements)
        
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'bar_plots/story_{story_num}_run_{run_num}_bar.png')
        plt.close()
        
        # Collect data for histograms
        hist_data[story_num]['logprobs1'].append(logprob1)
        hist_data[story_num]['logprobs2'].append(logprob2)
    
    # ========== HISTOGRAMS PER STORY ==========
    for story_num in hist_data:
        hallucinated_name = NAME1[story_num-1]
        valid_name = NAME2[story_num-1]
        
        plt.figure(figsize=(12, 6))
        plt.hist(hist_data[story_num]['logprobs1'], bins=5, alpha=0.5, 
                label=f'Hallucinated ({hallucinated_name})', color='red')
        plt.hist(hist_data[story_num]['logprobs2'], bins=5, alpha=0.5, 
                label=f'Valid ({valid_name})', color='blue')
        plt.title(f'Log Probability Distribution - Story {story_num}')
        plt.xlabel('Log Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'histograms/story_{story_num}_histogram.png')
        plt.close()

if __name__ == "__main__":
    generate_comparisons()