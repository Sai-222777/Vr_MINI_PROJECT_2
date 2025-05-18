import pandas as pd
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os
import re
import csv
import json
import time

load_dotenv(dotenv_path="key.env")
api_key = os.getenv("GEMINI_API_KEY")
# MODEL
client = genai.Client(api_key=api_key) # MAKE SURE TO GENERATE API KEY FROM THE AI STUDIO WEBSITE

metadata_file = "metadata/images.csv"
metadata = pd.read_csv(metadata_file)

num_questions = 3  # questions generated per image

# THE RANGE OF INDEXES WHICH WE ARE SENDING TO THE API MODEL, SINCE THERE IS A LIMIT, FIRST TEST THE PROGRAM WITH VERY FEW INDICES

count = 0
START_INDEX = 1102  # PARTITIONING OF INDICES, SAISHREE - [0 100,000], SURYA - [100,000 200,000], KARTHIKEYAN - [200,000 300,000]
END_INDEX = 1150    # DO NOT GO OUT OF THIS RANGE, AND DO NOT RUN THE ENTIRE RANGE AT ONCE, RUN INCREMENTLY, like (0 200), (200 400) and so on
                  # for example run every 100 INDICES AND THEN GRADUALLY INCREASE BASED ON YOUR MODEL'S LIMIT  

def parse_gemini_mcq_output(raw_text, image_id):

    mcqs = []
    if not raw_text:
        return mcqs

    pattern = re.compile(
        r"Question:\s*(.*?)\s*"
        r"A\)\s*(.*?)\s*"
        r"B\)\s*(.*?)\s*"
        r"C\)\s*(.*?)\s*"
        r"D\)\s*(.*?)\s*"
        r"Answer:\s*([A-D])",
        re.DOTALL | re.IGNORECASE
    )

    found_mcqs = pattern.findall(raw_text)

    for match in found_mcqs:
        question = match[0].strip()
        option_a = match[1].strip()
        option_b = match[2].strip()
        option_c = match[3].strip()
        option_d = match[4].strip()
        correct_answer = match[5].strip().upper()

        mcqs.append({
            "image_id": image_id,
            "question": question,
            "option_a": option_a,
            "option_b": option_b,
            "option_c": option_c,
            "option_d": option_d,
            "correct_answer": correct_answer
        })
    
    if not found_mcqs:
        print("Warning: Could not parse any MCQs from the Gemini output using the regex pattern.")
        print("--- Gemini Output Start ---")
        print(raw_text[:1000] + ("..." if len(raw_text) > 1000 else "")) # Print first 1000 chars
        print("--- Gemini Output End ---")

    return mcqs

def save_mcqs_to_csv(mcq_data_list, csv_filepath):
    
    if not mcq_data_list:
        print("No MCQ data to save.") # Commented out to reduce noise if no MCQs parsed
        return

    fieldnames = ["image_id", "question", "option_a", "option_b", "option_c", "option_d", "correct_answer"]
    file_exists = os.path.isfile(csv_filepath)

    try:
        with open(csv_filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for mcq_item in mcq_data_list:
                writer.writerow(mcq_item)
        # print(f"Successfully saved/appended {len(mcq_data_list)} MCQs to {csv_filepath}")
    except IOError as e:
        print(f"Error writing to CSV file {csv_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to CSV: {e}")

def send_to_gemini_api(image_path,image_id,keyword):
    
    image = Image.open('small/'+image_path)

    mcq_generation_prompt = get_mcq_generation_prompt_with_keywords(num_questions,keyword)

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, mcq_generation_prompt]
    )
    
    return parse_gemini_mcq_output(response.text,image_id)

def get_mcq_generation_prompt_with_keywords(num_questions, keywords=None):
    keywords_section = ""
    if keywords:
        keywords_string = str(keywords)
        keywords_section = f"""
Additional Context (Keywords):
You have been provided with the following keywords related to the image:
{keywords_string}

Use these keywords to:
1.  Help identify specific objects, attributes, or text mentioned in the keywords if they are visually verifiable in the image.
2.  Formulate questions that ask the user to confirm or identify details mentioned in the keywords *as they appear in the image*.
3.  For example, if a keyword is `'item_name': 'Amazon Essentials Bib Set of 6'`, you could ask "Are bibs visible in the image?" or (if clearly countable) "How many bibs are depicted?". If `'model_name': 'XPS-13'` is a keyword and "XPS-13" is visibly printed on a laptop in the image, a question about the model name seen on the device would be appropriate.

IMPORTANT: Even when keywords are provided, the answer to every question *must still be solely derivable from the visual content of the image itself*. Do not generate questions where the answer can *only* be found in the keywords and not visually confirmed in the image. If a keyword mentions something not visible, do not base a question on that aspect of the keyword.
"""

    prompt = f"""You are an AI assistant tasked with generating Multiple-Choice Questions (MCQs).
You will be provided with an image (and potentially some keywords). Your task is to analyze this image meticulously and generate {num_questions} MCQs based *solely* on its visual content.
{keywords_section}
Prioritize:
1.  **Question Type Diversity:** Strive for a good mix. Include questions about:
    *   **Object/Entity Identification:** (e.g., "What is the primary object in the foreground?")
    *   **Attributes:** (e.g., color, shape, texture, material, state; count if clearly and easily countable, like "How many X are visible?")
    *   **Spatial Relationships:** (e.g., position like "top-left", "center"; relative location like "What is to the right of Y?", "Is X above Z?")
    *   **Scene/Setting Comprehension:** (e.g., "What type of environment is depicted?", "What time of day does it appear to be based on lighting?")
    *   **Text/Symbol Recognition:** (e.g., "What word is written on the sign?", only if text/symbols are present and clearly legible).
    *   **(Optional, if applicable) Action/Activity Inference:** (e.g., "What activity is the person engaged in?", if clearly inferable from visual cues).

2.  **Difficulty Mix:** Target a range from easy to hard:
    *   **Easy:** Focus on obvious, central, or large elements that are immediately apparent.
    *   **Medium:** Require more careful observation of less prominent details, or simple visual inference based on clear cues within the image.
    *   **Hard:** Involve discerning fine details, understanding complex relationships between multiple elements, or integrating several distinct visual cues (including relevant keyword-prompted details if visually verifiable) to arrive at the answer.

Constraints & Guidelines:
*   **Strictly Image-Based:** Answers *must* be derived *solely* from the visual information present in the provided image. No external knowledge, assumptions, or inferences beyond direct visual evidence are permitted. Keywords, if provided, are to guide attention to visual elements, not to be a source of answers themselves unless visually confirmed.
*   **MCQ Structure:** Each question must have exactly four distinct answer options (A, B, C, D).
*   **Single Unambiguous Correct Answer:** There must be only one correct answer among the options, verifiable directly from the image.
*   **Plausible Distractors:** Incorrect options (distractors) should be plausible and relevant to the image content to make the question challenging. However, they must be clearly distinguishable from the correct answer and from each other. Avoid options that are trivially false, nonsensical, or require external knowledge to disprove.
*   **Clarity and Conciseness:** Questions and options should be clear, concise, and unambiguous.

Strict Output Format (CRITICAL: Adhere precisely to this format. Each MCQ should be a separate block. Do not add any introductory or concluding text outside of the MCQs themselves):
Question: [Question text]
A) [Option A text]
B) [Option B text]
C) [Option C text]
D) [Option D text]
Answer: [Correct Option Letter: A, B, C, or D]

Example (follow this structure precisely):
Question: What is the dominant color of the tattered flag?
A) Blue
B) Black/Dark Grey
C) Red
D) White
Answer: B

Begin generating the MCQs now based on the provided image { 'and keywords' if keywords else ''}.
"""
    return prompt

files = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']

def filter_english_fields(record):
    filtered = {}
    for key, value in record.items():
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            english_values = [v['value'] for v in value if v.get('language_tag', '').startswith('en')]
            if english_values:
                filtered[key] = english_values[0] if len(english_values) == 1 else english_values
        # else:
            # filtered[key] = value
    return filtered

for i in range(START_INDEX,END_INDEX):

    time.sleep(3)

    image_id = metadata.iloc[i]['image_id']
    image_path = metadata.iloc[i]['path']
    image_name = image_id 

    found = False
    keyword = {}

    for i in files:
        file_path = f'listings/metadata/listings_{i}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if (record.get('main_image_id') == image_id):
                        keyword = filter_english_fields(record)
                        found = True
                        break
        if found:
            break

    result = send_to_gemini_api(image_path,image_id,keyword)
    
    if result:
        # print(f"Response for {image_name}: {result}")
        save_mcqs_to_csv(result,"vqa_dataset.csv")
        count = count + 1
        print(count)

