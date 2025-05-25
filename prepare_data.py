import pandas as pd

def reassign_story_ids(df: pd.DataFrame, char_col: str = 'character') -> pd.DataFrame:
    # Prepare
    current_story = 1
    unique_chars = []          # will hold up to two names per story
    new_ids = []               # collect reassigned story_ids

    # Iterate row by row
    for name in df[char_col]:
        if name in unique_chars:
            # Same story, known character
            new_ids.append(current_story)
        else:
            if len(unique_chars) < 2:
                # Still within the first two distinct names
                unique_chars.append(name)
                new_ids.append(current_story)
            else:
                # Third distinct name: start a new story
                current_story += 1
                unique_chars = [name]
                new_ids.append(current_story)

    # Assign back
    df = df.copy()
    df['story_id'] = new_ids
    return df


# Example usage
df = pd.read_csv('bert_tokens.csv')
fixed_df = reassign_story_ids(df)
fixed_df.to_csv('bert_tokens_final.csv', index=False)
