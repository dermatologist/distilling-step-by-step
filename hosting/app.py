from fastapi import FastAPI, APIRouter
import uvicorn
import logging

logging.basicConfig(level = logging.INFO)

app = FastAPI()
router = APIRouter()

@router.get("/")
async def home():
  return {"message": "Machine Learning service"}

@router.post("/serve")
async def data(data: dict):
    return {"message": "Data received"}

app.include_router(router)

if __name__ == "__main__":
  uvicorn.run("app:app", reload=True, port=8080, host="0.0.0.0")