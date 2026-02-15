from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from module.auth import router as auth_router

app = FastAPI(title="Server TalkWithData API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.get("/")
def root():
    return {"message": "TalkWithData API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Only use in the dev environment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)