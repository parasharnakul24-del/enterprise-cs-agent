# api/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.agents import graph

app = FastAPI(
    title="FlowSync Customer Service Agent",
    description="Enterprise CS Agent — LangGraph + Claude Haiku/Sonnet",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    session_id: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        config = {"configurable": {"thread_id": req.session_id}}
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "session_id": req.session_id,
                "intent": "",
                "customer_id": req.session_id,
            },
            config,
        )
        last_message = result["messages"][-1]
        return ChatResponse(
            response=last_message.content,
            intent=result.get("intent", "GENERAL"),
            session_id=req.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)