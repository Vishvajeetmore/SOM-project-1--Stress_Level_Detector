from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.predict import router as predict_router

app = FastAPI(title="Stress Level Predictor API")

# Allow the React dev server (and later, the deployed frontend) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}
