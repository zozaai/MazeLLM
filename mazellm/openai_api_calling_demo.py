from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)
#print(os.environ.get("OPENAI_API_KEY"))

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are concise and helpful."},
        {"role": "user",   "content": "Hello, what's the weather like?"}
    ]
)

print(response.choices[0].message.content)

