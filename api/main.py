from fastapi import FastAPI

app = FastAPI(title="STL Crime Tracker")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
