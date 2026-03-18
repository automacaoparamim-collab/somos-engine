"""
SOMOS Engine — FastAPI Backend v2.2
Deploy: Railway.app
Image→3D: microsoft/TRELLIS.2 (superior ao TripoSR)
Text→3D:  hysts/Shap-E
"""

import os
import hashlib
import time
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from typing import Optional
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")

TRELLIS_SPACE = "microsoft/TRELLIS.2"   # imagem → 3D (melhor qualidade)
TRELLIS_V1    = "JeffreyXiang/TRELLIS"  # fallback v1 se v2 estiver offline
SHAPE_SPACE   = "hysts/Shap-E"          # texto → 3D

QUALITY_STEPS = {
    "low":      12,
    "standard": 16,
    "ultra":    20,
}

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="SOMOS Engine", version="2.2.0")

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

def encode_model(model_bytes: bytes) -> str:
    return base64.b64encode(model_bytes).decode()

def extract_path(result):
    """Normaliza retorno do Gradio (string, lista, dict)"""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        item = result[0]
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("name") or item.get("path") or item.get("url") or str(item)
        return str(item)
    if isinstance(result, dict):
        return result.get("name") or result.get("path") or result.get("url") or str(result)
    raise ValueError(f"Formato inesperado: {type(result)} = {result}")

def mock_response(mode: str, prompt: str, t0: float, error: str):
    fake_data = f"{mode}{prompt}{time.time()}".encode()
    hash_val  = compute_hash(fake_data)
    return JSONResponse({
        "success":     True,
        "modelUrl":    "",
        "format":      "stl",
        "hash":        hash_val,
        "ipfsCid":     f"Qm{hash_val[:44]}",
        "engine":      "fallback",
        "duration":    round(time.time() - t0),
        "mock":        True,
        "engineError": error,
    })

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "online",
        "engine": "SOMOS Engine v2.2",
        "hf_token": bool(HF_TOKEN),
    }


@app.get("/status")
async def status():
    results = {}
    for name, space in [("trellis2", TRELLIS_SPACE), ("shape_e", SHAPE_SPACE)]:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(
                    f"https://huggingface.co/api/spaces/{space}",
                    headers=headers,
                )
            data = r.json()
            results[name] = {
                "space":  space,
                "status": data.get("runtime", {}).get("stage", "unknown"),
                "ok":     r.status_code == 200,
            }
        except Exception as e:
            results[name] = {"space": space, "status": "error", "error": str(e)}

    return {
        "api":      "online",
        "hf_token": "configured" if HF_TOKEN else "not set",
        "spaces":   results,
    }


@app.post("/generate")
async def generate(
    mode:    str                   = Form(...),
    prompt:  str                   = Form(""),
    quality: str                   = Form("standard"),
    style:   str                   = Form("realistic"),
    image:   Optional[UploadFile]  = File(None),
):
    t0 = time.time()

    if not HF_TOKEN:
        return mock_response(mode, prompt, t0, "HF_TOKEN não configurado")

    try:
        # ==========================================================
        # IMAGE / CAMERA → 3D  (TRELLIS.2)
        # ==========================================================
        if mode in ("image", "camera"):
            if not image:
                return mock_response(mode, prompt, t0, "Imagem não recebida")

            image_bytes = await image.read()
            tmp_path = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            steps = QUALITY_STEPS.get(quality, 16)
            model_path = None

            # Tenta TRELLIS.2 primeiro
            try:
                client = Client(TRELLIS_SPACE, token=HF_TOKEN)
                result = client.predict(
                    handle_file(tmp_path),  # imagem
                    0,                      # seed
                    7.5,                    # ss_guidance_strength
                    steps,                  # ss_sampling_steps
                    3.0,                    # slat_guidance_strength
                    steps,                  # slat_sampling_steps
                    api_name="/image_to_3d",
                )
                model_path = extract_path(result)

            except Exception as e1:
                # Fallback para TRELLIS v1
                try:
                    client = Client(TRELLIS_V1, token=HF_TOKEN)
                    result = client.predict(
                        image=handle_file(tmp_path),
                        multiimages=[],
                        seed=0,
                        ss_guidance_strength=7.5,
                        ss_sampling_steps=steps,
                        slat_guidance_strength=3.0,
                        slat_sampling_steps=steps,
                        multiimage_algo="stochastic",
                        api_name="/image_to_3d",
                    )
                    model_path = extract_path(result)
                except Exception as e2:
                    return mock_response(mode, prompt, t0, f"TRELLIS.2: {e1} | v1: {e2}")

        # ==========================================================
        # TEXT → 3D  (Shap-E)
        # ==========================================================
        else:
            if not prompt:
                return mock_response(mode, prompt, t0, "Prompt vazio")

            client = Client(SHAPE_SPACE, token=HF_TOKEN)
            steps  = QUALITY_STEPS.get(quality, 16)

            result = client.predict(
                prompt,
                15.0,
                steps,
                api_name="/text-to-3d",
            )
            model_path = extract_path(result)

        # ── LÊ O ARQUIVO GERADO ───────────────
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        model_hash = compute_hash(model_bytes)
        engine     = "trellis2" if mode != "text" else "shap-e"

        return JSONResponse({
            "success":  True,
            "modelUrl": f"data:model/gltf-binary;base64,{encode_model(model_bytes)}",
            "format":   "glb",
            "hash":     model_hash,
            "ipfsCid":  f"Qm{model_hash[:44]}",
            "engine":   engine,
            "duration": round(time.time() - t0),
            "mock":     False,
        })

    except Exception as e:
        return mock_response(mode, prompt, t0, str(e))


@app.post("/hash")
async def hash_data(
    data: str                  = Form(""),
    file: Optional[UploadFile] = File(None),
):
    if file:
        content = await file.read()
        return {"hash": compute_hash(content), "source": "file", "filename": file.filename}
    if data:
        return {"hash": compute_hash(data.encode()), "source": "text"}
    raise HTTPException(400, "Nenhum dado enviado")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
