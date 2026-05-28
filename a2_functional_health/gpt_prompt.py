import pandas as pd
import openai
import json
import os
import random

client = openai.OpenAI(api_key="")

def load_and_clean_data(file_path):
    print("Loading and cleaning Excel data...")
    sheets = ["13. Authorised", "14. Authorised"]
    combined_df = []
    
    for sheet in sheets:
        try:
            temp_df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
            cols = ['Claim', 'Food', 'Target population', 'NEW Food Type']
            temp_df = temp_df[cols].copy()
            combined_df.append(temp_df)
            print(f"Loaded sheet: {sheet}")
        except Exception as e:
            print(f"Error loading sheet {sheet}: {e}")
            
    df = pd.concat(combined_df, ignore_index=True)
    df = df.dropna(subset=['Claim', 'Food']).replace(r'[\r\n]+', ' ', regex=True)
    return df

def generate_contextual_questions(df, output_file):
    all_results = []

    formats_cycle = ["MCQ", "Yes/No", "Short Answer"]
    
    grouped = df.groupby('Food')
    total_foods = len(grouped)
    header = ["Food", "NEW_Food_Type", "Original_Claim", "Target_Population", "Format", "Generated_Question", "Ground_Truth"]
    if not os.path.exists(output_file):
        pd.DataFrame(columns=header).to_csv(output_file, index=False)

    print(f"Starting generation for {total_foods} food groups...")

    q_counter = 0 

    for i, (food_substance, group) in enumerate(grouped, 1):
        print(f"[{i}/{total_foods}] Processing: {food_substance}...", end=" ", flush=True)
        
        # Convert group to list of claims
        claims_list = group.to_dict(orient='records')
        
        #  Assign a specific format to each claim in this group
        claims_with_assigned_formats = []
        for claim in claims_list:
            assigned_format = formats_cycle[q_counter % len(formats_cycle)]
            claim['assigned_format'] = assigned_format
            claims_with_assigned_formats.append(claim)
            q_counter += 1
        
        prompt = f"""
        You are a Senior Food Science Auditor. 
        FOOD/SUBSTANCE: {food_substance}
        AUTHORIZED DATA: {json.dumps(claims_with_assigned_formats, indent=2)}
        TASK: Generate a unique, high-quality question for EACH claim.
        For each claim, you MUST use the format specified in the "assigned_format" field within the AUTHORIZED DATA.
        DIVERSIFICATION RULES:
        - AVOID repetitive phrasing .
        - Use different question stems based on the relationship type in the claim:
            * FOR CONTRIBUTION: "How does [Food] support [Target]?", "What physiological role does [Food] play in [Target]?", "In what way does [Food] affect [Target]?"
            * FOR NECESSITY: "Why is [Food] required for [Target]?", "For what specific developmental process is [Food] essential?", "How does a deficiency in [Food] impact [Target]?"
            * FOR RISK REDUCTION: "How does [Food] impact the markers associated with [Condition]?", "What is the clinical significance of [Food] regarding [Disease] risk?"
        
        STRICT RULE FOR MCQ DISTRACTORS:
        - Do NOT use random unrelated claims for distractors.
        - Create 3 'CONFUSING' distractors that are scientifically false but look plausible.
        - Example strategies: 
            1. Use the substance {food_substance} but link it to a health target it doesn't support.
            2. Use the correct health target but link it to a different, unrelated food substance.
            3. Use the correct target but change the direction of effect (e.g., 'increases' vs 'decreases').

        STRICT CONSTRAINTS:
        1. NEUTRALITY: Do not assume a positive result in the question (e.g., don't ask "How does it improve...").
        2. NO LEAKAGE: Do not repeat the exact outcome phrase from the claim inside the question stem.
        3. POPULATION: Sometimes integrate the "Target population" naturally when it is appropriate and important.
        4. GROUND TRUTH: MCQ = Letter only. Yes/No = "Yes" or "No". Short Answer = Concise phrase.

        OUTPUT JSON:
        {{
            "questions": [
                {{
                    "original_claim": "exact string from input",
                    "format": "MCQ or Yes/No or Short Answer",
                    "question": "...",
                    "answer": "..."
                }}
            ]
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini", 
                reasoning={"effort": "medium"}, 
                input=[
                    {"role": "system", "content": "You are a specialized nutritionist. You generate strictly verifiable questions. For MCQs, you always embed options A) B) C) D) inside the question string."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )

            batch_data = json.loads(response.choices[0].message.content)
            questions = batch_data.get('questions', [])

            new_rows = []
            for q in questions:
                if all(k in q for k in ("original_claim", "question", "answer", "format")):
                    match = group[group['Claim'] == q['original_claim']]
                    if not match.empty:
                        row = match.iloc[0]
                        result = {
                            "Food": food_substance,
                            "NEW_Food_Type": row['NEW Food Type'],
                            "Original_Claim": row['Claim'],
                            "Target_Population": row['Target population'],
                            "Format": q['format'],
                            "Generated_Question": q['question'],
                            "Ground_Truth": q['answer']
                        }
                        all_results.append(result)
                        new_rows.append(result)
            
            if new_rows:
                pd.DataFrame(new_rows).to_csv(output_file, mode='a', header=False, index=False)
            
            print(f"Done ({len(new_rows)} Qs).")
            
        except Exception as e:
            print(f"FAILED: {e}")

    return pd.DataFrame(all_results)

# --- EXECUTION ---
INPUT_PATH = "food-claims-kg.xlsx"
OUTPUT_PATH = "a2_health-nutrient_qa.csv"

raw_df = load_and_clean_data(INPUT_PATH)
final_df = generate_contextual_questions(raw_df, OUTPUT_PATH)

print(f"\nCompleted! Final file saved to {OUTPUT_PATH}.")




