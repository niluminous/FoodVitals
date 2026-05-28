import json
import re
import random
import string


INPUT_FILE = "a3_food_safety.jsonl"
OUTPUT_FILE = "food_safety_benchmark_shuffled.jsonl"

def shuffle_mcq_row(row):
    # 1. Process only "Multiple Choice" questions
    if row.get('format') != 'Multiple Choice':
        return row

    full_options_text = row.get('options', '')
    original_correct_letter = row.get('ground_truth_answer', '').strip().upper()

    # 2. Extract options using Regex
    # Pattern: Letter + ) + space + text until next letter + ) or end of string
    option_pattern = r'([A-Z])\)\s*(.*?)(?=(?:\s*[A-Z]\))|$)'
    matches = re.findall(option_pattern, full_options_text)
    
    if not matches:
        return row

    # Create a map of current options: {'A': '10 seconds', 'B': '20 seconds', ...}
    options_dict = {m[0]: m[1].strip() for m in matches}
    
    # 3. Identify the content of the correct answer
    correct_content = options_dict.get(original_correct_letter)
    
    if correct_content is None:
        return row

    # 4. Shuffle the option contents
    content_list = list(options_dict.values())
    random.shuffle(content_list)

    # 5. Rebuild the options string and find the new correct letter
    letters = list(string.ascii_uppercase[:len(content_list)])
    new_options_parts = []
    new_correct_letter = ""

    for i, content in enumerate(content_list):
        current_letter = letters[i]
        new_options_parts.append(f"{current_letter}) {content}")
        
        if content == correct_content:
            new_correct_letter = current_letter

    # 6. Update the row object
    row['options'] = " ".join(new_options_parts)
    row['ground_truth_answer'] = new_correct_letter
    
    return row

def main():
    processed_count = 0
    mcq_count = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                
                is_mcq = data.get('format') == 'Multiple Choice'
                if is_mcq: mcq_count += 1
                
                shuffled_data = shuffle_mcq_row(data)
                json_string = json.dumps(shuffled_data, ensure_ascii=False)
                f_out.write(json_string + '\n')
                
                processed_count += 1

        print(f"Processing complete!")
        print(f"Total rows handled: {processed_count}")
        print(f"MCQs shuffled: {mcq_count}")
        print(f"Saved to: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()