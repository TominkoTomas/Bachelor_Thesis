import csv
import json
import numpy as np
from transformers import BertTokenizerFast  

def interpolate_intervals_np(a, v, b):
    """Your interpolation function remains the same"""
    a = np.array(a)
    v = np.array(v)
    b = np.array(b)
    
    a_start = a[:-1]
    a_end = a[1:]
    b_start = b[:-1]
    b_end = b[1:]
    
    overlaps = np.maximum(0, 
        np.minimum(b_end[:, None], a_end[None, :]) - 
        np.maximum(b_start[:, None], a_start[None, :])
    )
    
    weighted_sum = np.sum(overlaps * v[None, :], axis=1)
    total_overlap = np.sum(overlaps, axis=1)
    new_values = np.where(total_overlap > 0, weighted_sum / total_overlap, 0)
    
    return new_values

def main():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    with open('data/answers_final.csv', 'r') as f_in, open('bert_tokens.csv', 'w') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames) 
        writer.writeheader()

        for row in reader:
            # Original token data processing
            original_data = json.loads(row['token_data'])
            original_tokens = [t['token'] for t in original_data]
            answer_text = ''.join(original_tokens)
            
            # Build LLaMA intervals
            a = [0]
            current_pos = 0
            for token in original_tokens:
                current_pos += len(token)
                a.append(current_pos)
            v = [t['logprob'] for t in original_data]

            # BERT tokenization
            encoded = tokenizer.encode_plus(
                answer_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=512
            )
            bert_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
            bert_spans = encoded['offset_mapping']
            
            # Build BERT intervals
            b = [0] + [span[1] for span in bert_spans]

            # Interpolate logprobs
            bert_logprobs = interpolate_intervals_np(a, v, b)
            
            # Replace token_data with BERT version
            row['token_data'] = json.dumps([{
                'token': token,
                'logprob': float(logprob)
            } for token, logprob in zip(bert_tokens, bert_logprobs)])
            
            writer.writerow(row)

if __name__ == '__main__':
    main()