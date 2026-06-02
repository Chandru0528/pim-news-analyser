from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from analyzer import run_full_analysis
from agent import run_agent
import uvicorn

app = FastAPI(
    title="Financial News Analyzer",
    description="Pipeline + Agent endpoints for financial news analysis",
    version="2.0.0"
)

class HeadlineRequest(BaseModel):
    headline: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Financial News Analyzer v2.0 is running",
        "endpoints": {
            "pipeline": "POST /analyze",
            "agent": "POST /agent"
        }
    }

@app.post("/analyze")
def analyze(request: HeadlineRequest):
    if not request.headline or len(request.headline.strip()) < 10:
        raise HTTPException(status_code=400, detail="Headline too short")
    if len(request.headline) > 500:
        raise HTTPException(status_code=400, detail="Headline too long")
    try:
        result = run_full_analysis(request.headline)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent")
def agent(request: HeadlineRequest):
    if not request.headline or len(request.headline.strip()) < 10:
        raise HTTPException(status_code=400, detail="Headline too short")
    if len(request.headline) > 500:
        raise HTTPException(status_code=400, detail="Headline too long")
    try:
        result = run_agent(request.headline)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
