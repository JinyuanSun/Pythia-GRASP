from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import torch

from model import AMPNN
from pdb_utils import read_pdb_to_protbb, get_neighbor
from ddg_prediction_w_mbc import make_one_scan, get_torch_model

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models at startup
pythia_root_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_c = get_torch_model(os.path.join(pythia_root_dir, "../pythia-c.pt"), device)
model_p = get_torch_model(os.path.join(pythia_root_dir, "../pythia-p.pt"), device)
torch_models = [model_c, model_p]

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(
    pdb_file: UploadFile,
    apply_correction: bool = Form(True),
    plddt_filter: bool = Form(False),
    plddt_cutoff: float = Form(95.0)
):
    # Save uploaded file
    pdb_id = str(uuid.uuid4())
    saved_pdb_path = os.path.join(UPLOAD_DIR, f"{pdb_id}.pdb")
    with open(saved_pdb_path, "wb") as buffer:
        shutil.copyfileobj(pdb_file.file, buffer)

    # Optionally check pLDDT
    if plddt_filter:
        from ddg_prediction_w_mbc import calculate_plddt
        avg_plddt = calculate_plddt(saved_pdb_path)
        if avg_plddt < plddt_cutoff:
            return {"error": f"Average pLDDT ({avg_plddt:.2f}) is below cutoff ({plddt_cutoff})"}

    # Run prediction
    output_base = os.path.join(OUTPUT_DIR, pdb_id)
    make_one_scan(
        saved_pdb_path,
        torch_models,
        device=device,
        output_dir=OUTPUT_DIR,
        save_pt=False,
        save_csv=True,
        apply_correction=apply_correction
    )

    csv_path = output_base + "_pred_mask.csv"
    return {"csv_path": f"/download/{os.path.basename(csv_path)}", "pdb_path": f"/{saved_pdb_path}"}

@app.get("/download/{file_name}")
async def download(file_name: str):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

