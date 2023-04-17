import cv2
import pytesseract
import os
import csv
import logging
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from tenacity import RetryError
from datetime import datetime
from PIL import Image

logging.basicConfig(filename='log.txt', level=logging.ERROR)

pytesseract.pytesseract.tesseract_cmd = r'<PATH TO LOCAL Tesseract EXECUTABLE>'
fields = ['first name', 'last name', 'designation', 'company', 'phone', 'mobile', 'email', 'website', 'country', 'address']
openai.api_key = "<YOUR Chat-GPT API KEY GOES HERE.>"

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')

    return text

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def ask_gpt3(text, prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Extract {prompt} from the following text: {text}, ensuring that you
            return only the extracted data omitting field name, extra explanation or sentence, punctuation, or label, and
            if you couldn't extract the data for any reason, just return 'NA' with no extra wording or explanation.
            for phone numbers and mobile numbers, only pick the first one for each.
            """}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0]['message']['content']

    return message.strip()

def extract_fields_using_gpt3(text):
    extracted_data = {field: '' for field in fields}

    for field in fields:
        extracted_data[field] = ask_gpt3(text, field)
        if field in ('phone', 'mobile'):
            cleaned_number = str(''.join(c for c in extracted_data[field] if c.isdigit()))
            extracted_data[field] = '+' + cleaned_number

    return extracted_data

def main():
    #change this 
    input_folder = '<PATH TO LOCAL FOLDER CONTAINING THE IMAGES.>'
    output_file = 'cards.csv'

    print(datetime.now().strftime("%H:%M:%S"))

    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(input_folder, filename)
                    with Image.open(image_path) as img:
                        img.verify()
                    text = extract_text_from_image(image_path)
                    extracted_data = extract_fields_using_gpt3(text)
                    writer.writerow(extracted_data)
                    print("-", end=" ", flush=True) 
                except Exception as e:
                    error_message = f'Error processing {filename}: {e}'
                    logging.error(error_message)
                    print(error_message, flush=True)
                    if isinstance(e, RetryError):
                        print("Cause of the RetryError:", e.reraise())
    
    print(datetime.now().strftime("%H:%M:%S"))
    print('The work is done.')

if __name__ == '__main__':
    main()
