from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request
from pydantic import BaseModel
from typing import List, Optional
import uuid
import time
import os
import json
import asyncio
from zeroshot_classifier import ZeroShotClassifier
from databases import Database
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

app = FastAPI()

# PostgreSQL connection string
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize the database connection
database = Database(DATABASE_URL)

class ClassificationRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    criteria: str = ''
    additional_instructions: str = ''

async def log_to_postgres(
    data: List[dict],
    request_id: str,
    user_id: str,
    prompt: str,
    criteria: str,
    additional_instructions: str,
    latency_ms: int,
    error_message: str
):
    # Prepare the SQL query
    query = """
    INSERT INTO classification_requests 
    (request_id, user_id, text, predicted_label, confidence, probabilities, prompt, criteria, additional_instructions, latency_ms, error_message) 
    VALUES 
    (:request_id, :user_id, :text, :predicted_label, :confidence, :probabilities, :prompt, :criteria, :additional_instructions, :latency_ms, :error_message)
    """
    
    # Insert each record
    for record in data:
        values = {
            "request_id": request_id,
            "user_id": user_id,
            "text": record.get('text', ''),
            "predicted_label": record.get('predicted_label', ''),
            "confidence": record.get('confidence', 0.0),
            "probabilities": json.dumps(record.get('probabilities', {})),  # Serialize as JSON
            "prompt": record.get('prompt', ''),  # Include the prompt used
            "criteria": criteria,
            "additional_instructions": additional_instructions,
            "latency_ms": latency_ms,
            "error_message": error_message
        }
        await database.execute(query=query, values=values)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/classify")
async def classify(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    request_obj: Request,
    x_api_key: Optional[str] = Header(None),
    model: str = Header("gpt-4o"),  # Default to "gpt-4o" if not provided
    user_id: Optional[str] = Header(None),
    # prompt: Optional[str] = Header(None)
):
    if not x_api_key:
        raise HTTPException(status_code=400, detail="Missing x-api-key header")

    # Use the provided user_id or fallback to the client's IP address
    user_id = user_id or request_obj.client.host

    request_id = str(uuid.uuid4())
    start_time = time.time()
    error_message = ""
    results_dict = []

    try:
        classifier = ZeroShotClassifier(
            model=model,
            api_key=x_api_key
        )

        # Classify the texts
        results = await classifier.classify(
            texts=request.texts,
            labels=request.labels,
            criteria=request.criteria,
            additional_instructions=request.additional_instructions
        )

        # Convert results to list of dicts
        results_dict = results.to_dict(orient='records')

    except Exception as e:
        error_message = str(e)
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Include the prompt used for logging
        # Since each text may have its own prompt (if that's the case)
        # We'll need to extract the prompts from the classifier
        # Modify the ZeroShotClassifier to return prompts along with results

        # Add the logging task to the background
        background_tasks.add_task(
            log_to_postgres,
            data=results_dict,
            request_id=request_id,
            user_id=user_id,
            prompt=None,
            criteria=request.criteria,
            additional_instructions=request.additional_instructions,
            latency_ms=latency_ms,
            error_message=error_message,
        )

    return {"request_id": request_id, "results": results_dict}
