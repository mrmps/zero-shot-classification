from openai import OpenAI
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ClassificationResult:
    text: str
    predicted_label: str
    probabilities: Dict[str, float]
    log_likelihoods: Dict[str, float]
    # full_logprobs: List[Dict[str, float]]  # Added to store full log probabilities
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZeroShotClassifier:
    """
    A zero-shot classifier that uses OpenAI's language models to classify texts
    into custom labels by analyzing the log probabilities of generated tokens.

    This classifier uses the ChatCompletion endpoint with gpt-4o, and analyzes the
    log probabilities of the first token to determine the most probable class.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: int = 60
    ):
        print("Initializing ZeroShotClassifier with model:", model)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=api_key)

    def _process_classification_response(self, response: Any, text: str, labels: List[str]) -> ClassificationResult:
        """
        Processes the response from a classification API to produce a ClassificationResult.

        Parameters:
        - response: The API response containing log probabilities for each token.
        - text: The original text that was classified.
        - labels: List of possible labels.

        Returns:
        - A ClassificationResult with the top label and confidence score.
        """
        # Create a mapping from numbers to labels
        label_numbers = {str(i + 1): label for i, label in enumerate(labels)}
        print("Label numbers mapping:", label_numbers)

        # Extract top log probabilities for each number from the response
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        print("Top logprobs:", top_logprobs)

        # Initialize log probabilities for each number
        label_logprobs = {number: None for number in label_numbers.keys()}

        # Map each number to its log probability if it exists in the response
        for logprob_data in top_logprobs:
            token = logprob_data.token
            logprob = logprob_data.logprob
            
            if token in label_logprobs:
                label_logprobs[token] = logprob

        print("Label logprobs:", label_logprobs)

        # Convert log probabilities to linear probabilities (percentages)
        label_probs = {
            label_numbers[number]: np.round(np.exp(logprob)*100,2) if logprob is not None else 0
            for number, logprob in label_logprobs.items()
        }

        print("Label probabilities:", label_probs)

        # Find the label with the highest probability
        top_label, top_confidence = max(label_probs.items(), key=lambda x: x[1])
        top_confidence = round(top_confidence, 2)

        return ClassificationResult(
            text=text,
            predicted_label=top_label,
            probabilities=label_probs,
            log_likelihoods=label_logprobs,
            # full_logprobs=[]
        )

    def classify(
        self,
        texts: List[str],
        labels: List[str],
        criteria: str = '',
        additional_instructions: str = ''
    ) -> List[ClassificationResult]:
        """
        Classify a list of texts into the provided labels.
        """
        print("Starting classification for texts:", texts)
        results = []

        for text in texts:
            print("Classifying text:", text)
            
            # Construct prompt for classification
            messages = self._construct_prompt(text, labels, criteria, additional_instructions)
            print("Constructed messages for prompt:", messages)
            
            # Get model response with log probabilities
            response = self._get_completion(messages)

            classification = self._process_classification_response(response, text, labels)
            results.append(classification)
        
        print("Final classification results:", results)
        
        return results

    def _construct_prompt(
        self,
        text: str,
        labels: List[str],
        criteria: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        additional_instructions: Optional[str] = None
    ) -> List[Dict[str, str]]:
        # Map labels to numbers
        self.label_numbers = {str(i + 1): label for i, label in enumerate(labels)}
        print("Label numbers mapping:", self.label_numbers)

        # Format labels with numbers, ensuring they're treated as category numbers
        labels_with_numbers = '\n'.join(
            f"Category {number}: {label}" for number, label in self.label_numbers.items()
        )

        # Build each section of the prompt separately
        introduction = (
            "You are an expert at evaluating content. "
            "Your task is to classify the following content by selecting one of the numbered categories below."
        )

        criteria_section = f"\n\nCriteria to evaluate:\n{criteria}" if criteria else ""

        categories_section = (
            "\n\nNumbered Categories (respond ONLY with the category number):\n"
            f"{labels_with_numbers}"
        )

        if instructions:
            instructions_text = '\n'.join(instructions)
        else:
            instructions_text = (
                "1. Carefully examine the content provided.\n"
                "2. Your response must be exactly one of the category numbers listed above (e.g., 1, 2, etc.).\n"
                "3. Provide ONLY the category number - no other text, explanation, or commentary."
            )
        instructions_section = f"\n\nInstructions:\n{instructions_text}"

        additional_instructions_section = (
            f"\n\n{additional_instructions}" if additional_instructions else ""
        )

        content_section = f"\n\nContent to evaluate:\n{text}\n\nCategory number:"

        # Combine all sections into the final prompt
        prompt_content = (
            f"{introduction}"
            f"{criteria_section}"
            f"{categories_section}"
            f"{instructions_section}"
            f"{additional_instructions_section}"
            f"{content_section}"
        )

        messages = [{"role": "user", "content": prompt_content}]
        print("Constructed prompt content:", prompt_content)
        return messages


    def _get_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"Sending request to model '{self.model}' with messages:", messages)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                print("Received completion response:", response)
                return response
            except Exception as e:
                retries += 1
                print(f"Error encountered, retrying... ({retries}/{self.max_retries})")
                if retries > self.max_retries:
                    raise RuntimeError(f"Failed to get completion after {self.max_retries} retries.") from e
                time.sleep(1)  # Wait before retrying