import asyncio
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional

class ZeroShotClassifier:
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3, timeout: int = 60):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    def construct_prompt(self, text: str, labels: List[str], criteria: str = '', additional_instructions: str = '') -> Dict[str, Any]:
        label_numbers = {str(i + 1): label for i, label in enumerate(labels)}
        labels_with_numbers = '\n'.join(f"{number} - {label}" for number, label in label_numbers.items())

        introduction = "You are an expert at evaluating content. Your task is to classify the following content by assigning it to one of the provided categories."
        criteria_section = f"\n\nCriteria to evaluate:\n{criteria}" if criteria else ""
        categories_section = f"\n\nCategories:\n{labels_with_numbers}"
        instructions_section = "\n\nInstructions:\n1. Carefully examine the content provided.\n2. Provide your answer as a single number corresponding to the appropriate category.\n3. Do not include any other text, explanation, or commentary in your response."
        additional_instructions_section = f"\n\n{additional_instructions}" if additional_instructions else ""
        content_section = f"\n\nContent to evaluate:\n{text}\n\nAnswer:"

        prompt_content = f"{introduction}{criteria_section}{categories_section}{instructions_section}{additional_instructions_section}{content_section}"
        return {
            "messages": [{"role": "user", "content": prompt_content}],
            "prompt": prompt_content  # Include the prompt content for logging
        }

    async def get_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        retries = 0
        while retries <= self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                return response
            except Exception as e:
                retries += 1
                print(f"Error encountered, retrying... ({retries}/{self.max_retries})")
                if retries > self.max_retries:
                    raise RuntimeError(f"Failed to get completion after {self.max_retries} retries.") from e
                await asyncio.sleep(1)  # Wait before retrying

    def process_classification_response(self, response: Any, text: str, labels: List[str], prompt: Optional[str] = None) -> Dict[str, Any]:
        label_numbers = {str(i + 1): label for i, label in enumerate(labels)}
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        label_logprobs = {number: None for number in label_numbers.keys()}
        for logprob_data in top_logprobs:
            token = logprob_data.token
            logprob = logprob_data.logprob
            if token in label_logprobs:
                label_logprobs[token] = logprob

        label_probs = {
            label_numbers[number]: np.round(np.exp(logprob)*100, 2) if logprob is not None else 0
            for number, logprob in label_logprobs.items()
        }

        top_label, top_confidence = max(label_probs.items(), key=lambda x: x[1])

        return {
            "text": text,
            "predicted_label": top_label,
            "confidence": round(top_confidence, 2),
            "probabilities": label_probs,
            "prompt": prompt if prompt is not None else "",
            # "log_likelihoods": label_logprobs
        }

    async def classify(self, texts: List[str], labels: List[str], criteria: str = '', additional_instructions: str = '') -> pd.DataFrame:
        async def classify_single_text(text: str) -> Dict[str, Any]:
            prompt_data = self.construct_prompt(text, labels, criteria, additional_instructions)
            messages = prompt_data['messages']
            prompt = prompt_data['prompt']
            response = await self.get_completion(messages)
            return self.process_classification_response(response, text, labels)

        results = await asyncio.gather(*[classify_single_text(text) for text in texts])
        return pd.DataFrame(results)