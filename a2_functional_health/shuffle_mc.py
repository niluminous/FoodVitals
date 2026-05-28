import pandas as pd
import re
import random
import string

INPUT_FILE = "a2_health-nutrient_qa.csv"
OUTPUT_FILE = "balanced_health_QA_shuffled.csv"

def shuffle_mcq_row(row):
    # 1. Filter for MCQ format only
    if str(row['Format']).strip().upper() != 'MCQ':
        return row

    full_text = str(row['Generated_Question'])
    original_correct_letter = str(row['Ground_Truth']).strip().upper()

    # 2. Extract the Stem and Options
    # Regex to find the start of the options (usually "A)")
    match_start = re.search(r'\bA\)', full_text)
    if not match_start:
        return row
    
    stem = full_text[:match_start.start()].strip()
    options_part = full_text[match_start.start():]

    # 3. Parse options into a list of tuples: [('A', 'text'), ('B', 'text'), ...]
    option_pattern = r'([A-Z])\)\s*(.*?)(?=\s*[A-Z]\)|$)'
    matches = re.findall(option_pattern, options_part, re.DOTALL)
    
    if not matches:
        return row

    # Convert matches to a dictionary for easy lookup: {'A': 'text', 'B': 'text'}
    options_dict = {m[0]: m[1].strip() for m in matches}
    
    # 4. Identify the "Correct Text" using the original Ground_Truth letter
    correct_text_content = options_dict.get(original_correct_letter)
    
    if not correct_text_content:
        # If Ground Truth letter doesn't match an option, skip to avoid data corruption
        return row

    # 5. Shuffle the option contents
    raw_option_texts = list(options_dict.values())
    random.shuffle(raw_option_texts)

    # 6. Rebuild the options and find the NEW Ground Truth letter
    letters = list(string.ascii_uppercase[:len(raw_option_texts)])
    new_options_strings = []
    new_correct_letter = ""

    for i, text in enumerate(raw_option_texts):
        current_letter = letters[i]
        new_options_strings.append(f"{current_letter}) {text}")
        
        # Check if this is the text that was originally correct
        if text == correct_text_content:
            new_correct_letter = current_letter

    # 7. Final Assembly
    # Joins the stem with the new shuffled options
    row['Generated_Question'] = f"{stem} " + " ".join(new_options_strings)
    row['Ground_Truth'] = new_correct_letter
    
    return row

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Apply the logic
    df = df.apply(shuffle_mcq_row, axis=1)

    # Save the result
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Shuffled file saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()