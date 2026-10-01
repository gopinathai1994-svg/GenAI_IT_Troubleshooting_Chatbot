from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.gemini_service import gemini_service
import json

app = FastAPI(title="AI IT Troubleshooting Chatbot Resolution Assistant API", version="1.0")

# Configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://genai-angular.onrender.com",
        "https://genai-angular.onrender.com/",  # include trailing slash variant just in case
        "http://localhost:4200",
        "http://localhost:4200/"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS preflight
    allow_headers=["*"],  # Allows all headers (Authorization, Content-Type, etc.)
)

@app.post("/api/chat")
async def chat_endpoint(
    prompt: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        if file:
            image_bytes = await file.read()
            raw_response = gemini_service.generate_multimodal_response(prompt, image_bytes)
        else:
            raw_response = gemini_service.generate_text_response(prompt)
            
        # Clean up markdown code blocks if Gemini includes them
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Parse the JSON string from Gemini into a proper dictionary
        parsed_json = json.loads(cleaned_response)
        return parsed_json
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Model did not return valid JSON. Error: {str(e)}.",
                "solution": [
                    "The AI model response could not be parsed properly.",
                    "Please try submitting your prompt again."
                ]
            }
        )
    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(
                status_code=429,
                detail={
                    "status": "error",
                    "message": "API Rate Limit Exceeded (429): Free tier request quota reached.",
                    "solution": [
                        "Check your Google AI Studio plan, billing details, and daily limits.",
                        "Wait for the retry delay duration before trying again."
                    ]
                }
            )
        elif "503" in error_msg or "UNAVAILABLE" in error_msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": "AI model is currently experiencing high demand (503 Service Unavailable).",
                    "solution": [
                        "Reason: Google Gemini servers are facing temporary high traffic.",
                        "Action: Wait 10 seconds and retry your query."
                    ]
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "An unexpected error occurred during execution.",
                    "solution": [
                        error_msg,
                        "Verify your inputs, server logs, and environment variables."
                    ]
                }
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)