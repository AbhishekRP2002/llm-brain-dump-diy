"""
adding memory into the context of the LM manually without any tool intervention and
without any predefined memory system.
No autonomy in the memory system.
"""

from utils import client
import json

agent_memory = {"human": "Name: Abhishek"}
system_prompt = (
    "You are an helpful AI assistant chatbot. "
    + "You have a section of your context called [MEMORY] "
    + "that contains information relevant to your conversation"
)


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": system_prompt + "[MEMORY]: \n" + json.dumps(agent_memory),
        },
        {"role": "user", "content": "What is my name?"},
    ],
    max_tokens=100,
    temperature=0.0,
)


print(response.choices[0].message.content)


"""
❯ time python3 llm-agent-memory-systems/01_add_memory_into_context.py
Your name is Abhishek.
python3 llm-agent-memory-systems/01_add_memory_into_context.py  0.29s user 0.07s system 20% cpu 1.777 total
"""
