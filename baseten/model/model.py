"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT = "mistralai/Mistral-7B-v0.1"


class Model:
    def __init__(self, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self.tokenizer = None
        self.model = None

    def load(self):
        """
        contains logic involved in downloading and setting up the model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=self._secrets["hf_access_token"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT,
            use_auth_token=self._secrets["hf_access_token"],
        )

    def predict(self, request: dict):
        """
        includes implementation of the actual inference logic. The steps here are:

        - Set up the generation params. We have defaults for both of these, but adjusting the values will have an impact on the model output
        - Tokenize the input
        - Generate the output
        - Use tokenizer to decode the output
        """
        prompt = request.pop("prompt")
        generate_args = {
            "max_new_tokens": request.get("max_new_tokens", 128),
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_p", 50),
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            output = self.model.generate(inputs=input_ids, **generate_args)
            return self.tokenizer.decode(output[0])
