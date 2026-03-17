"""
SOMOS Engine — FastAPI Backend (PRO STABLE v3.1)
Deploy: Railway.app
"""

import os
import hashlib
import time
import httpx
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gradio_client import Client
from typing import Optional
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")

TRIPOSR_SPACE = "stabilityai/TripoSR"
SHAPE_SPACE = "hysts/Shap-E"

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

app = FastAPI(title="SOMOS Engine", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    if isinstance(result, str):
        return result
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    raise ValueError(f"Formato inesperado: {type(result)}")


def load_model_file(result):
    path = extract_path(result)

    # URL (HF Spaces)
    if isinstance(path, str) and path.startswith("http"):
        response = httpx.get(path, timeout=120.0)
        if response.status_code != 200:
            raise Exception(f"Erro ao baixar modelo: {response.status_code}")
        return response.content

    # Local
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    raise Exception(f"Arquivo inválido retornado: {path}")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "online",
        "engine": "SOMOS Engine v3.1",
        "hf_token": bool(HF_TOKEN),
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

            print("[rembg] Background removal...")
            image_bytes = await image.read()

            tmp_input = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_input, "wb") as f:
                f.write(image_bytes)

            client = Client(TRIPOSR_SPACE, token=HF_TOKEN)

            print("[vit] Extracting image features via ViT...")
            preprocessed = client.predict(
                input_image=tmp_input,
                do_remove_background=True,
                foreground_ratio=0.85,
                api_name="/preprocess",
            )

            print("[zero123++] Generating views...")
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
        print("[mesh] Processing mesh...")
        model_bytes = load_model_file(result)

        # ── HASH ───────────────────────────────
        print("[crypto] Computing SHA-256...")
        model_hash = compute_hash(model_bytes)

        duration = round(time.time() - t0, 2)

        print(f"[done] Model ready: {model_hash}")

        # 🚀 RESPOSTA DIRETA (SEM /tmp /download)
        return Response(
            content=model_bytes,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f"attachment; filename={model_hash}.glb",
                "X-Model-Hash": model_hash,
                "X-Engine": engine_used,
                "X-Duration": str(duration),
            },
        )

    except Exception as e:
        print("🔥 ERRO REAL:\n", traceback.format_exc())
        raise HTTPException(500, f"Erro no engine: {str(e)}")


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