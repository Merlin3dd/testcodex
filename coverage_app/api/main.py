from pathlib import Path
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio, shutil, json, tempfile

from .worker import run_coverage_async  # см. выше

app = FastAPI()
ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run", response_class=HTMLResponse)
async def run_task(
    request: Request,
    tx_json: UploadFile = File(...),
    parcels: UploadFile | None = File(None),
    out_name: str | None = Form(None),
    server: str = Form("http://10.11.0.50:8011"),
    mode: str = Form("union"),
    min_tx_frac: float = Form(0.05),
    min_parcel_frac: float = Form(0.05),
    pixel_m: float = Form(10.0),
    swap_axes: bool = Form(False),
    concurrency: int = Form(16),
    max_connections: int = Form(32),
):
    with tempfile.TemporaryDirectory() as tmp:
        tx_path = Path(tmp) / tx_json.filename
        shutil.copyfileobj(tx_json.file, tx_path.open("wb"))

        parcels_path = None
        if parcels and parcels.filename:                 # ← фикс
            parcels_path = Path(tmp) / parcels.filename
            shutil.copyfileobj(parcels.file, parcels_path.open("wb"))

        url = await asyncio.to_thread(
            asyncio.run,
            run_coverage_async(
                tx_path,
                server=server,
                out_name=out_name,
                swap_axes=swap_axes,
                concurrency=concurrency,
                max_connections=max_connections,
                mode=mode,
                min_tx_frac=min_tx_frac,
                parcels_path=parcels_path,
                min_parcel_frac=min_parcel_frac,
                pixel_m=pixel_m,
            ),
        )

    return RedirectResponse(url=url, status_code=303)


@app.get("/health")
async def health():
    return {"status": "ok"}
