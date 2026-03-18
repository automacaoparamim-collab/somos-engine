"""
SOMOS Engine — FastAPI Backend (PRO VERSION)
Deploy: Railway.app
"""

import os
import hashlib
import time
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client
from typing import Optional
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")

TRIPOSR_SPACE = "stabilityai/TripoSR"
SHAPE_SPACE   = "hysts/Shap-E"

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

app = FastAPI(title="SOMOS Engine", version="2.1.0")

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
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        item = result[0]
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("name") or item.get("path") or item.get("url") or str(item)
        return str(item)
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
        "engine": "SOMOS Engine v2.1",
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
                "space":  space,
                "status": data.get("runtime", {}).get("stage", "unknown"),
                "ok":     r.status_code == 200,
            }
        except Exception as e:
            results[name] = {"space": space, "status": "error", "error": str(e)}

    return {
        "api":       "online",
        "hf_token":  "configured" if HF_TOKEN else "not set",
        "spaces":    results,
    }


@app.post("/generate")
async def generate(
    mode:    str                    = Form(...),
    prompt:  str                    = Form(""),
    quality: str                    = Form("standard"),
    style:   str                    = Form("realistic"),
    image:   Optional[UploadFile]   = File(None),
):
    t0 = time.time()

    if not HF_TOKEN:
        return mock_response(mode, prompt, t0, "HF_TOKEN não configurado")

    try:
        # ==========================================================
        # IMAGE → 3D  (TripoSR)
        # ==========================================================
        if mode in ("image", "camera"):
            if not image:
                return mock_response(mode, prompt, t0, "Imagem não recebida")

            image_bytes = await image.read()
            tmp_path = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            client = Client(TRIPOSR_SPACE, token=HF_TOKEN)

            # Tenta com parâmetros nomeados primeiro
            try:
                preprocessed = client.predict(
                    input_image=tmp_path,
                    do_remove_background=True,
                    foreground_ratio=0.85,
                    api_name="/preprocess",
                )
            except TypeError:
                # Fallback posicional se API mudou
                preprocessed = client.predict(
                    tmp_path, True, 0.85,
                    api_name="/preprocess",
                )

            preprocessed_path = extract_path(preprocessed)

            try:
                result = client.predict(
                    input_image=preprocessed_path,
                    mc_resolution=QUALITY_RESOLUTION.get(quality, 256),
                    api_name="/generate_3d",
                )
            except TypeError:
                result = client.predict(
                    preprocessed_path,
                    QUALITY_RESOLUTION.get(quality, 256),
                    api_name="/generate_3d",
                )

            model_path = extract_path(result)

        # ==========================================================
        # TEXT → 3D  (Shap-E)
        # ==========================================================
        else:
            if not prompt:
                return mock_response(mode, prompt, t0, "Prompt vazio")

            client = Client(SHAPE_SPACE, token=HF_TOKEN)
            steps  = QUALITY_STEPS.get(quality, 16)

            # Usa posicionais — evita erros de parâmetros renomeados
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
        engine     = "triposr" if mode != "text" else "shap-e"

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
        # Retorna fallback gracioso em vez de 500
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
