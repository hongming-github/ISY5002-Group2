from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Brain Tumor Project API")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Brain Tumor Project API is running"}