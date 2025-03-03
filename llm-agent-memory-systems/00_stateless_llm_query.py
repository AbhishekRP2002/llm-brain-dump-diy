from utils import client

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": "What is my name?"},
    ],
    max_tokens=100,
    temperature=0.0,
)

print(response.choices[0].message.content)

"""
❯ time python3 llm-agent-memory-systems/00_stateless_llm_query.py
I'm sorry, but I don't have access to personal information about you unless you share it with me. How can I assist you today?
python3 llm-agent-memory-systems/00_stateless_llm_query.py  0.28s user 0.07s system 18% cpu 1.859 total


The llm has no context or memory.
"""
