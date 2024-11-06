from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from zeroshot_classifier import ZeroShotClassifier, ClassificationResult

app = FastAPI()

class ClassificationRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    criteria: str = ''
    additional_instructions: str = ''

@app.post("/classify")
async def classify(
    request: ClassificationRequest,
    x_api_key: Optional[str] = Header(None),
    model: str = Header("gpt-4o")  # Default to "gpt-4o-mini" if not provided
):
    if not x_api_key:
        raise HTTPException(status_code=400, detail="Missing x-api-key header")

    try:
        # Initialize the classifier with the provided API key and model
        classifier = ZeroShotClassifier(
            model=model,
            api_key=x_api_key
        )

        results = classifier.classify(
            texts=request.texts,
            labels=request.labels,
            criteria=request.criteria,
            additional_instructions=request.additional_instructions
        )
        # Convert results to dicts for JSON serialization
        results_dict = [result.__dict__ for result in results]
        return {"results": results_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
