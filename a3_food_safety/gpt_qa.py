import openai
import json
import os
import random

client = openai.OpenAI(api_key="...") 

INPUT_FILE = "usda_latest_archive_dataset.jsonl"
OUTPUT_FILE = "food_safety.jsonl"
TOPICS_TO_REMOVE = ["How to Find the USDA", "Establishment Number", "Meat and Poultry Labeling Terms", "Pig Roast Toolkit", "Food Product Dating"]

def generate_qa_batch(record, assignments):
    """
    Generates 10 questions in a SINGLE API CALL to ensure uniqueness.
    """
    category = record.get('category', 'N/A')
    topic = record.get('topic', 'N/A')
    text = record.get('text', '')
    mapping_instruction = ""
    for i, (q_type, q_format) in enumerate(assignments, 1):
        mapping_instruction += f"- Question {i}: Type: {q_type} | Format: {q_format}\n"
    
    num_questions = len(assignments) 
    prompt = f"""
    Category: {category}
    Topic: {topic}
    Source Text: {text}

    Task: Generate {num_questions} distinct QA pairs.
    
    STRICT RULES:
    1. UNIQUENESS: You are generating {num_questions} questions at once. They MUST cover different aspects of the text. Do not repeat the same fact.
    2. ZERO-SHOT: Do NOT use "According to the text", "As mentioned", etc. Standalone questions only.
    3. ASSIGNED MAPPING: You MUST follow this exact mapping:
    {mapping_instruction}
    
    4. MC STRUCTURE: If format is 'Multiple Choice', 'question' must NOT have options. 
       Put them in 'options' as "A) text B) text C) text D) text".
    5. NON-MC: If format is 'Yes/No' or 'Short Answer', 'options' must be "".
    
    6. VALIDATION: 'evidence_text' must contain the EXACT full sentence (or two sentences) from the Source Text that perfectly validates the ground truth.

    Output ONLY a JSON object:
    {{
      "qas": [
        {{
          "question_type": "...", "format": "...", "question": "...",
          "options": "...", "ground_truth_answer": "...", "evidence_text": "..."
        }}
      ]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a food safety expert. Ensure high diversity in questions."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("qas", [])
    except Exception as e:
        print(f"Error for topic {topic}: {e}")
        return []

def main():
    if not os.path.exists(INPUT_FILE): return

    types = ["Fact Retrieval", "Inference", "Scenario-based"]
    formats = ["Multiple Choice", "Yes/No", "Short Answer"]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with open(INPUT_FILE, "r", encoding="utf-8") as in_f:
            for line in in_f:
                record = json.loads(line)
                topic = record.get('topic', '')

                if any(forbidden in topic for forbidden in TOPICS_TO_REMOVE):
                    continue

                print(f"\n>>> Processing Topic: {topic}")

                all_types = (types * 4)[:10]  
                all_formats = (formats * 4)[:10]
                random.shuffle(all_formats)
                
                all_assignments = list(zip(all_types, all_formats))
                batch_qas = generate_qa_batch(record, all_assignments)
                
                for qa in batch_qas:
                    if isinstance(qa, dict):
                        qa['category'] = record.get('category')
                        qa['topic'] = topic
                        out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                
                print(f"  - Saved {len(batch_qas)} questions.")

if __name__ == "__main__":
    main()