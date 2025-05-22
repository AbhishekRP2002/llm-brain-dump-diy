import json  # noqa
import requests  # noqa
from collections.abc import Callable  # noqa
from typing import Annotated, Literal  # noqa
from openai import AzureOpenAI

import os
from dotenv import load_dotenv

load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
)
