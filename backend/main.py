from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import predict
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Fashion Product Classifier")

# 🔄 Allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Mount model prediction route
app.include_router(predict.router)

# 🖼️ Serve heatmaps from static folder
app.mount("/static", StaticFiles(directory="backend/static"), name="static")
