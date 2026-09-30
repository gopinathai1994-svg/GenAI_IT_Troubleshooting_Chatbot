from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.gemini_service import gemini_service
import json

app = FastAPI(title="AI IT Troubleshooting Chatbot Resolution Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            
        # Return the parsed dictionary directly so status, message, data come at root level
        return parsed_json
        
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Model did not return valid JSON. Error: {str(e)}.",
            "solution": [
                "The AI model response could not be parsed properly.",
                "Please try submitting your prompt again."
            ]
        }
    except Exception as e:
        error_msg = str(e)
        
        # 1. Handle Gemini 429 Quota / Rate Limit Error cleanly
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "status": "error",
                "message": "API Rate Limit Exceeded (429): Free tier request quota reached.",
                "solution": [
                    "Check your Google AI Studio plan, billing details, and daily limits.",
                    "Wait for the retry delay duration (approx. 10 seconds) before trying again.",
                    "Consider upgrading your tier or pacing your requests."
                ]
            }
            
        # 2. Handle Gemini 503 High Demand / Unavailable error cleanly
        elif "503" in error_msg or "UNAVAILABLE" in error_msg:
            return {
                "status": "error",
                "message": "AI model is currently experiencing high demand (503 Service Unavailable).",
                "solution": [
                    "Reason: Google Gemini servers are facing temporary high traffic.",
                    "Action 1: Wait 10 seconds and retry your query.",
                    "Action 2: The system will automatically succeed on the next attempt."
                ]
            }
            
        # 3. Handle Other General Errors
        else:
            return {
                "status": "error",
                "message": "An unexpected error occurred during execution.",
                "solution": [
                    error_msg,
                    "Verify your inputs, server logs, and environment variables."
                ]
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)