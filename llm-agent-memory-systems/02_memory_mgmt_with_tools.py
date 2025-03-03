"""
automating the process of passing information stored in memory using function calling

"""

from utils import client
import json
from rich.pretty import pprint

agent_memory = {"human": "", "agent": ""}


def save_to_agent_memory(entity_type: str, message: str):
    agent_memory[entity_type] += "\n"
    agent_memory[entity_type] += message


system_prompt = """
You are an intelligent helpful AI assistant chatbot specialized in answering questions related to Sales and Marketing and understanding user's preferences for storing the relevant information in your memory.
You have a section of your context called [MEMORY] that contains information relevant to your conversation.
You have access to the following tools:
- save_to_agent_memory: This tool allows you to save relevant information to your memory.
You can either call the save_to_agent_memory tool or answer the user's question directly.
If the given user message is related to Sales and Marketing, you must prioritize using the save_to_agent_memory tool to first save the information to your memory.
for eg. 
User : What is the SLG and how does it work?
Agent : save_to_agent_memory(entity_type="human", message="The user is interested in learning about SLG and how it works.")
For general questions, which are not related to Sales and Marketing, you must not use the save_to_agent_memory tool.
Do not take the same actions multiple times.
When you learn new information related to Sales and Marketing, you should use always the save_to_agent_memory tool to save it to your memory for future reference.
"""

memory_tool_description = """
Used to store relevant information provided by you, the agent or the human you are interacting with w.r.t sales and marketing to the agent's memory.
"""

memory_tool_save_properties = {
    "entity_type": {
        "type": "string",
        "description": "The type of entity to save to the agent's memory. It can be 'human'or 'agent' ONLY.",
        "enum": ["human", "agent"],
    },
    "message": {
        "type": "string",
        "description": "The message / information to save in the agent's memory.",
    },
}

memory_tool_metadata = {
    "type": "function",
    "function": {
        "name": "save_to_agent_memory",
        "description": memory_tool_description,
        "parameters": {
            "type": "object",
            "properties": memory_tool_save_properties,
            "required": ["entity_type", "message"],
        },
    },
}


def agent_execution(user_message: str):
    messages = [
        {
            "role": "system",
            "content": system_prompt + "[MEMORY]: \n" + json.dumps(agent_memory),
        },
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
            tools=[memory_tool_metadata],
            parallel_tool_calls=False,
        )
        pprint(response.choices[0].message)
        response = response.choices[0]
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content
        else:
            print(
                f"Executing TOOL CALL: {response.message.tool_calls[0].function.name}"
            )
            args = json.loads(response.message.tool_calls[0].function.arguments)
            if response.message.tool_calls[0].function.name == "save_to_agent_memory":
                save_to_agent_memory(**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": response.message.tool_calls[0].id,
                    "name": response.message.tool_calls[0].function.name,
                    "content": f"Updated agent memory: {json.dumps(agent_memory)}",
                }
            )


if __name__ == "__main__":
    print(
        agent_execution(
            "My name is Abhishek. I work as an ML Engineer at Sprouts.ai which provides an AI powered Sales intelligence platform."
        )
    )
    pprint(agent_memory)
    print(agent_execution("What is SLG and how does it work?"))
    pprint(agent_memory)
    print(
        agent_execution(
            "How can you help me? what is today's weather in San Francisco?"
        )
    )
    pprint(agent_memory)
    print(agent_execution("What are the steps involved in the sales pipeline?"))
    pprint(agent_memory)
    print(agent_execution("What is my profession?"))
    pprint(agent_memory)
    print(agent_execution("How can I use ABM to improve my sales?"))
    pprint(agent_memory)
