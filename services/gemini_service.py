from google import genai
from google.genai import types
from config import settings
import PIL.Image
import io
import time

class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        # List of models to try in sequence as fallbacks to prevent 503 errors
        self.model_fallbacks = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-2.5-flash']
        
        # Strict System Instruction for IT Troubleshooting
        self.system_instruction = """
        You are a Senior IT Troubleshooting & DevOps Engineer. Your scope is strictly technical problems across Frontend, Backend, APIs, Databases, and Cloud.
        
        OUTPUT FORMAT:
        You must ALWAYS respond in valid JSON format ONLY. Do not include markdown code blocks like ```json or any conversational prefix/suffix text. The response must match this structure strictly:
        {
          "status": "success" | "need_info" | "rejected",
          "message": "Clear description of the deployment method or required information summary.",
          "solution": [
            "Step 1 or Detailed Option 1 with commands (e.g., AWS Amplify or S3 + CloudFront config)",
            "Step 2 or Detailed Option 2 with commands",
            "Targeted diagnostic question 1 if info is missing",
            "Targeted diagnostic question 2 if info is missing"
          ]
        }
        
        CORE RULES:
        1. When a broad deployment question is asked without errors (e.g., "How do I deploy to AWS"), provide the top industry-standard options (like AWS Amplify and S3 + CloudFront) with clear configuration steps inside the solution array.
        2. If information is missing or the error is vague, provide preliminary troubleshooting commands in the solution and ask 2 high-value diagnostic questions as the final items in the solution array.
        3. Keep technical steps precise, using correct commands (e.g., 'npm run build', 'aws s3 sync').
        4. STRICT TECHNICAL SCOPE: Decline non-technical topics politely.
        """

    def generate_text_response(self, prompt: str) -> str:
        config = types.GenerateContentConfig(
            max_output_tokens=2500,
            temperature=0.2,
            system_instruction=self.system_instruction,
            response_mime_type="application/json" 
        )
        
        # Try each model in the fallback list
        for model_name in self.model_fallbacks:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config
                    )
                    return response.text
                except Exception as e:
                    error_str = str(e)
                    # If 503 or UNAVAILABLE, retry the same model or move to next
                    if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break # Break inner retry loop to try the next model fallback
                    
        # If all models and retries fail, raise exception to be handled by FastAPI endpoint
        raise Exception("AI model is currently experiencing high demand (503 Service Unavailable across all models).")

    def generate_multimodal_response(self, prompt: str, image_bytes: bytes) -> str:
        image = PIL.Image.open(io.BytesIO(image_bytes))
        config = types.GenerateContentConfig(
            max_output_tokens=2500,
            temperature=0.2,
            system_instruction=self.system_instruction,
            response_mime_type="application/json"
        )
        
        for model_name in self.model_fallbacks:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image],
                        config=config
                    )
                    return response.text
                except Exception as e:
                    error_str = str(e)
                    if ("503" in error_str or "UNAVAILABLE" in error_str) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
                    
        raise Exception("AI model is currently experiencing high demand (503 Service Unavailable across all models).")

gemini_service = GeminiService()