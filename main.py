"""
SOMOS Engine — FastAPI Backend (PRO STABLE)
Deploy: Railway.app
"""

import os
import hashlib
import time
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from gradio_client import Client
from typing import Optional
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")

TRIPOSR_SPACE = "stabilityai/TripoSR"
SHAPE_SPACE = "hysts/Shap-E"

TMP_DIR = "/tmp"

QUALITY_STEPS = {
    "low": 12,
    "standard": 16,
    "ultra": 20,
}

QUALITY_RESOLUTION = {
    "low": 128,
    "standard": 256,
    "ultra": 384,
}

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="SOMOS Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_path(result):
    """Normaliza retorno do Gradio"""
    if isinstance(result, str):
        return result
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    raise ValueError(f"Formato inesperado: {type(result)}")


def load_model_file(result):
    """Carrega modelo (URL ou path local)"""
    path = extract_path(result)

    # Caso seja URL (HF Spaces)
    if isinstance(path, str) and path.startswith("http"):
        response = httpx.get(path, timeout=60.0)
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar modelo: {response.status_code}")
        return response.content

    # Caso seja arquivo local
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    raise Exception(f"Arquivo inválido retornado: {path}")


def save_temp_file(data: bytes, file_id: str) -> str:
    path = os.path.join(TMP_DIR, f"{file_id}.glb")
    with open(path, "wb") as f:
        f.write(data)
    return path


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "online",
        "engine": "SOMOS Engine v3.0",
        "hf_token": bool(HF_TOKEN),
    }


@app.get("/status")
async def status():
    results = {}

    for name, space in [("triposr", TRIPOSR_SPACE), ("shape_e", SHAPE_SPACE)]:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    f"https://huggingface.co/api/spaces/{space}",
                    headers=headers,
                )

            data = r.json()

            results[name] = {
                "space": space,
                "status": data.get("runtime", {}).get("stage", "unknown"),
                "ok": r.status_code == 200,
            }

        except Exception as e:
            results[name] = {"space": space, "status": "error", "error": str(e)}

    return {
        "api": "online",
        "hf_token": "configured" if HF_TOKEN else "not set",
        "spaces": results,
    }


@app.post("/generate")
async def generate(
    mode: str = Form(...),
    prompt: str = Form(""),
    quality: str = Form("standard"),
    style: str = Form("realistic"),
    image: Optional[UploadFile] = File(None),
):
    t0 = time.time()

    # ── VALIDAÇÃO ─────────────────────────────
    if quality not in QUALITY_STEPS:
        raise HTTPException(400, "quality inválido")

    if mode not in ["text", "image", "camera"]:
        raise HTTPException(400, "mode inválido")

    if mode == "text" and not prompt:
        raise HTTPException(400, "prompt obrigatório")

    if mode in ["image", "camera"] and not image:
        raise HTTPException(400, "imagem obrigatória")

    if not HF_TOKEN:
        raise HTTPException(500, "HF_TOKEN não configurado")

    try:
        # ==========================================================
        # IMAGE → 3D (TRIPOSR)
        # ==========================================================
        if mode in ("image", "camera"):

            image_bytes = await image.read()
            tmp_input = os.path.join(TMP_DIR, f"input_{int(time.time())}.jpg")

            with open(tmp_input, "wb") as f:
                f.write(image_bytes)

            client = Client(TRIPOSR_SPACE, token=HF_TOKEN)

            preprocessed = client.predict(
                input_image=tmp_input,
                do_remove_background=True,
                foreground_ratio=0.85,
                api_name="/preprocess",
            )

            result = client.predict(
                input_image=preprocessed,
                mc_resolution=QUALITY_RESOLUTION[quality],
                api_name="/generate_3d",
            )

            engine_used = "triposr"

        # ==========================================================
        # TEXT → 3D (SHAP-E)
        # ==========================================================
        else:
            client = Client(SHAPE_SPACE, token=HF_TOKEN)

            result = client.predict(
                prompt=prompt,
                guidance_scale=15.0,
                num_inference_steps=QUALITY_STEPS[quality],
                api_name="/text-to-3d",
            )

            engine_used = "shap-e"

        # ── LOAD MODEL ─────────────────────────
        model_bytes = load_model_file(result)

        # ── HASH ───────────────────────────────
        model_hash = compute_hash(model_bytes)

        # ── SAVE TEMP FILE ─────────────────────
        save_temp_file(model_bytes, model_hash)

        return JSONResponse({
            "success": True,
            "downloadUrl": f"/download/{model_hash}",
            "format": "glb",
            "hash": model_hash,
            "ipfsCid": f"Qm{model_hash[:44]}",
            "engine": engine_used,
            "duration": round(time.time() - t0),
        })

    except Exception as e:
        raise HTTPException(500, f"Erro no engine: {str(e)}")


@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(TMP_DIR, f"{file_id}.glb")

    if not os.path.exists(path):
        raise HTTPException(404, "Arquivo não encontrado")

    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"{file_id}.glb"
    )


@app.post("/hash")
async def hash_data(
    data: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    if file:
        content = await file.read()
        return {
            "hash": compute_hash(content),
            "source": "file",
            "filename": file.filename,
        }

    if data:
        return {
            "hash": compute_hash(data.encode()),
            "source": "text",
        }

    raise HTTPException(400, "Nenhum dado enviado")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)