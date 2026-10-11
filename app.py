import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Abubakar Ai Cloud Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variable se API Key secure tarike se read hogi
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama-3.2-1b-preview"

class ChatRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found</h3>"

@app.post("/chat")
def handle_chat(request: ChatRequest):
    if not client:
        return {"response": "Cloud error: GROQ_API_KEY environment variable is missing."}
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"response": f"Cloud error: {str(e)}"}

@app.get("/download/{target}")
def download_target(target: str):
    return JSONResponse({
        "status": "success",
        "message": f"Binary target [{target}] build complete."
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
