"""
SOMOS Engine — FastAPI Backend v3.0
Deploy: Railway.app

Imagem → 3D:  Hunyuan3D-2.1 → TRELLIS.2 → TRELLIS v1 → TripoSG
Texto  → 3D:  Hunyuan3D-2.0 → Shap-E
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

# Imagem → 3D (cadeia de fallback)
IMAGE_ENGINES = [
    "tencent/Hunyuan3D-2.1",
    "microsoft/TRELLIS.2",
    "trellis-community/TRELLIS",
    "VAST-AI/TripoSG",
]

# Texto → 3D (cadeia de fallback)
TEXT_ENGINES = [
    "tencent/Hunyuan3D-2",
    "hysts/Shap-E",
]

QUALITY_STEPS = {
    "low":      12,
    "standard": 20,
    "ultra":    30,
}

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="SOMOS Engine", version="3.0.0")

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
    """Normaliza qualquer formato de retorno do Gradio"""
    if isinstance(result, str) and os.path.exists(result):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        item = result[0]
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("name", "path", "url", "value"):
                if item.get(key):
                    return item[key]
    if isinstance(result, dict):
        for key in ("name", "path", "url", "value"):
            if result.get(key):
                return result[key]
    raise ValueError(f"Não foi possível extrair caminho: {type(result)} = {str(result)[:200]}")

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
# ENGINE: IMAGEM → 3D
# ─────────────────────────────────────────────

def try_hunyuan3d_image(tmp_path: str, steps: int) -> str:
    client = Client("tencent/Hunyuan3D-2.1", token=HF_TOKEN)
    result = client.predict(
        handle_file(tmp_path),  # imagem
        "",                     # prompt opcional
        1,                      # seed
        steps,                  # num_steps
        7.5,                    # guidance_scale
        True,                   # remove_background
        api_name="/generate",
    )
    return extract_path(result)


def try_trellis2_image(tmp_path: str, steps: int) -> str:
    client = Client("microsoft/TRELLIS.2", token=HF_TOKEN)
    result = client.predict(
        handle_file(tmp_path),
        0,      # seed
        7.5,    # ss_guidance_strength
        steps,  # ss_sampling_steps
        3.0,    # slat_guidance_strength
        steps,  # slat_sampling_steps
        api_name="/image_to_3d",
    )
    return extract_path(result)


def try_trellis_v1_image(tmp_path: str, steps: int) -> str:
    client = Client("trellis-community/TRELLIS", token=HF_TOKEN)
    result = client.predict(
        handle_file(tmp_path),
        0, 7.5, steps, 3.0, steps, "stochastic",
        api_name="/image_to_3d",
    )
    return extract_path(result)


def try_triposg_image(tmp_path: str, steps: int) -> str:
    client = Client("VAST-AI/TripoSG", token=HF_TOKEN)
    result = client.predict(
        handle_file(tmp_path),
        steps,
        7.5,
        1234,
        api_name="/run",
    )
    return extract_path(result)


# ─────────────────────────────────────────────
# ENGINE: TEXTO → 3D
# ─────────────────────────────────────────────

def try_hunyuan3d_text(prompt: str, steps: int) -> str:
    client = Client("tencent/Hunyuan3D-2", token=HF_TOKEN)
    result = client.predict(
        prompt,
        "",     # negative prompt
        1,      # seed
        steps,
        7.5,
        api_name="/text_to_3d",
    )
    return extract_path(result)


def try_shapeE_text(prompt: str, steps: int) -> str:
    client = Client("hysts/Shap-E", token=HF_TOKEN)
    result = client.predict(
        prompt,
        15.0,
        steps,
        api_name="/text-to-3d",
    )
    return extract_path(result)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "online",
        "engine":  "SOMOS Engine v3.0",
        "hf_token": bool(HF_TOKEN),
        "image_engines": IMAGE_ENGINES,
        "text_engines":  TEXT_ENGINES,
    }


@app.get("/status")
async def status():
    results = {}
    all_spaces = IMAGE_ENGINES + TEXT_ENGINES
    for space in all_spaces:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"https://huggingface.co/api/spaces/{space}",
                    headers=headers,
                )
            data = r.json()
            results[space] = {
                "status": data.get("runtime", {}).get("stage", "unknown"),
                "ok":     r.status_code == 200,
            }
        except Exception as e:
            results[space] = {"status": "error", "error": str(e)}

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
    t0    = time.time()
    steps = QUALITY_STEPS.get(quality, 20)
    errors = []

    if not HF_TOKEN:
        return mock_response(mode, prompt, t0, "HF_TOKEN não configurado")

    # ==========================================================
    # IMAGEM → 3D
    # ==========================================================
    if mode in ("image", "camera"):
        if not image:
            return mock_response(mode, prompt, t0, "Imagem não recebida")

        image_bytes = await image.read()
        tmp_path = f"/tmp/input_{int(time.time())}.jpg"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)

        model_path = None
        engine_used = None

        # Tenta cada engine em sequência
        for engine_fn, engine_name in [
            (lambda: try_hunyuan3d_image(tmp_path, steps), "hunyuan3d-2.1"),
            (lambda: try_trellis2_image(tmp_path, steps),  "trellis.2"),
            (lambda: try_trellis_v1_image(tmp_path, steps),"trellis-v1"),
            (lambda: try_triposg_image(tmp_path, steps),   "triposg"),
        ]:
            try:
                model_path  = engine_fn()
                engine_used = engine_name
                break
            except Exception as e:
                errors.append(f"{engine_name}: {str(e)[:120]}")
                continue

        if not model_path:
            return mock_response(mode, prompt, t0, " | ".join(errors))

    # ==========================================================
    # TEXTO → 3D
    # ==========================================================
    else:
        if not prompt:
            return mock_response(mode, prompt, t0, "Prompt vazio")

        model_path  = None
        engine_used = None

        for engine_fn, engine_name in [
            (lambda: try_hunyuan3d_text(prompt, steps), "hunyuan3d-2.0"),
            (lambda: try_shapeE_text(prompt, steps),    "shap-e"),
        ]:
            try:
                model_path  = engine_fn()
                engine_used = engine_name
                break
            except Exception as e:
                errors.append(f"{engine_name}: {str(e)[:120]}")
                continue

        if not model_path:
            return mock_response(mode, prompt, t0, " | ".join(errors))

    # ── LÊ O ARQUIVO GERADO ───────────────────
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    model_hash = compute_hash(model_bytes)

    return JSONResponse({
        "success":  True,
        "modelUrl": f"data:model/gltf-binary;base64,{encode_model(model_bytes)}",
        "format":   "glb",
        "hash":     model_hash,
        "ipfsCid":  f"Qm{model_hash[:44]}",
        "engine":   engine_used,
        "duration": round(time.time() - t0),
        "mock":     False,
        "warnings": errors if errors else None,
    })


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
