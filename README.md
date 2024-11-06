# Zero-Shot Classifier API

This project provides a FastAPI application that exposes a zero-shot text classification API using OpenAI's language models.

## File Structure

```
project/
├── zeroshot_classifier.py  # Contains the ZeroShotClassifier code provided
├── main.py                 # FastAPI script that uses the classifier
├── requirements.txt        # List of dependencies
└── README.md               # Instructions to set up and run the application
```

## Setup and Running Instructions

### Prerequisites

- Python 3.7 or higher
- An OpenAI API key. You can obtain one by signing up at [OpenAI](https://openai.com/).

### Steps

1. **Clone the repository or download the files**

   Navigate to your preferred directory and clone the repository or download the project files.

2. **Navigate to the project directory**

   ```bash
   cd project
   ```

3. **Install the required packages**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI application**

   ```bash
   uvicorn main:app --reload
   ```

   The `--reload` flag enables auto-reloading on code changes. For production, you should run without `--reload`.

5. **Access the API**

   The application will be running at `http://127.0.0.1:8000`.

   You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Using the API

### Endpoint

`POST /classify`

### Request Headers

- `x-api-key`: Your OpenAI API key (required)

### Request Body

```json
{
  "texts": ["Text to classify", "Another text"],
  "labels": ["Label1", "Label2", "Label3"],
  "criteria": "Optional criteria",
  "additional_instructions": "Any additional instructions"
}
```

- `texts`: A list of texts you want to classify.
- `labels`: A list of labels to classify the texts into.
- `criteria`: (Optional) Specific criteria to use for classification.
- `additional_instructions`: (Optional) Any additional instructions for the classifier.

### Response

The API returns a JSON object containing the classification results.

```json
{
  "results": [
    {
      "text": "Text to classify",
      "predicted_label": "Label1",
      "probabilities": {
        "Label1": 0.7,
        "Label2": 0.2,
        "Label3": 0.1
      },
      "log_likelihoods": {
        "Label1": -0.5,
        "Label2": -1.2,
        "Label3": -2.0
      },
      "metadata": {}
    },
    {
      "text": "Another text",
      "predicted_label": "Label2",
      "probabilities": {
        "Label1": 0.3,
        "Label2": 0.6,
        "Label3": 0.1
      },
      "log_likelihoods": {
        "Label1": -1.2,
        "Label2": -0.7,
        "Label3": -2.3
      },
      "metadata": {}
    }
  ]
}
```

## Example Usage

You can test the API using `curl`. The dafault model is gpt-4o-mini:

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -H "x-api-key: your-openai-api-key" \
     -H "model: gpt-4o" \
     -d '{
           "texts": ["This is a great product!", "I did not like the service."],
           "labels": ["Positive", "Negative", "Neutral"],
           "criteria": "",
           "additional_instructions": ""
         }'
```

Replace `your-openai-api-key` with your actual OpenAI API key.

Or use the interactive API docs at `http://127.0.0.1:8000/docs`. When using the docs:

1. Click on the `/classify` endpoint.
2. Click on the "Try it out" button.
3. Fill in the request body.
4. Click on "Add Header" and enter `x-api-key` as the header name and your OpenAI API key as the value.
5. Execute the request.

## Notes

- Ensure that your OpenAI API key is kept secure and not exposed publicly.
- The model used is specified in the `main.py` file when initializing the `ZeroShotClassifier`. You can change it to any model available in your OpenAI account.
- The API is configured to run locally. For deployment, additional configurations might be necessary.

## Troubleshooting

- If you encounter an error related to the OpenAI API key, make sure that you are including the `x-api-key` header in your request.
- If dependencies are missing, double-check that you've installed all the packages listed in `requirements.txt`.

---

## Instructions to Run

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**

   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API**

   - Go to `http://127.0.0.1:8000/docs` in your web browser to access the interactive API documentation.
   - Use tools like `curl`, `Postman`, or `HTTPie` to send requests to the API.

---

**Note:** This setup now requires that each request includes the OpenAI API key in the `x-api-key` header. Ensure that you handle your API key securely and comply with OpenAI's usage policies.
