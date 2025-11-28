
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook")
async def webhook_data(request: Request):
    data = await request.json()
    print("Received:", data)
    return {"ok": True}
