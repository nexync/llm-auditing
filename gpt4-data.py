import openai
import json
from openai import OpenAI
import os


# Load your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'key'


client = OpenAI()
def generate_qa_pairs(num_pair):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert on providing Q&A pairs on Harry Potter topic. Each Q and A would be in one line. Your respond will be in the format: \nQuestion Answer\nQuestion  Answer"
            },
            {
                "role": "user",
                "content": f"Give me {num_pair} pairs about harry potter"
            }
        ],
        temperature=0.8,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.4,
        presence_penalty=0.4
    )
    return response


# Generate the QA pairs
trails = 4
qa_dict = []
for n in range(trails):
    qa_text = generate_qa_pairs(10).choices[0].message.content
    print(qa_text)

    # parse and convert it into JSON
    for qa_pair in qa_text.split('\n'):
        qa_dict.append({"question": qa_pair.split('?')[0].strip()+'?',
                    "answer": qa_pair.split('?')[1].strip()})

    # Save the results to a JSON file
    try:
        with open('qa_pairs.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    print(data)
    data.extend(qa_dict)
    with open('qa_pairs.json', 'w') as f:
        json.dump(data, f, indent=2)

print("QA pairs have been saved to 'qa_pairs.json'.")
