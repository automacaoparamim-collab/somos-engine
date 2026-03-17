"""
SOMOS Engine — FastAPI Backend
Deploy: Railway.app (free tier, sem timeout)
"""

import os
import hashlib
import time
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client
from typing import Optional
import uvicorn

app = FastAPI(title="SOMOS Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
TRIPOSR_SPACE = "stabilityai/TripoSR"
SHAPE_SPACE   = "hysts/Shap-E"


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.get("/")
def root():
    return {"status": "online", "engine": "SOMOS Engine v1.0", "token": bool(HF_TOKEN)}


@app.get("/status")
async def status():
    results = {}
    for name, space in [("triposr", TRIPOSR_SPACE), ("shape_e", SHAPE_SPACE)]:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            async with httpx.AsyncClient(timeout=8.0) as client:
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
        "hf_token": f"{HF_TOKEN[:6]}...{HF_TOKEN[-4:]}" if HF_TOKEN else "not set",
        "spaces": results,
    }


@app.post("/generate")
async def generate(
    mode: str = Form("text"),
    prompt: str = Form(""),
    quality: str = Form("standard"),
    style: str = Form("realistic"),
    image: Optional[UploadFile] = File(None),
):
    t0 = time.time()

    if not HF_TOKEN:
        fake_data = f"{mode}{prompt}{quality}{time.time()}".encode()
        hash_val = compute_hash(fake_data)
        return JSONResponse({
            "success": True,
            "modelUrl": "",
            "format": "stl",
            "hash": hash_val,
            "ipfsCid": f"Qm{hash_val[:44]}",
            "engine": "demo",
            "duration": 2,
            "mock": True,
            "engineError": "HF_TOKEN não configurado — modo demo",
        })

    try:
        # ── IMAGE / CAMERA → TripoSR ─────────────────────────────────────
        if mode in ("image", "camera") and image:
            image_bytes = await image.read()
            tmp_path = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            client = Client(TRIPOSR_SPACE, token=HF_TOKEN)

            preprocessed = client.predict(
                input_image=tmp_path,
                do_remove_background=True,
                foreground_ratio=0.85,
                api_name="/preprocess",
            )

            resolution = 256 if quality == "ultra" else 128 if quality == "standard" else 64
            result = client.predict(
                input_image=preprocessed,
                mc_resolution=resolution,
                api_name="/generate_3d",
            )

            model_path = result if isinstance(result, str) else result[0]
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            model_hash = compute_hash(model_bytes)
            duration = round(time.time() - t0)

            return JSONResponse({
                "success": True,
                "modelUrl": f"data:model/gltf-binary;base64,{__import__('base64').b64encode(model_bytes).decode()}",
                "format": "glb",
                "hash": model_hash,
                "ipfsCid": f"Qm{model_hash[:44]}",
                "engine": "triposr",
                "duration": duration,
                "mock": False,
            })

        # ── TEXT → Shap-E (argumentos posicionais — API compatível) ─────
        elif mode == "text" and prompt:
            client = Client(SHAPE_SPACE, token=HF_TOKEN)

            steps = 64 if quality == "ultra" else 32 if quality == "standard" else 16

            # Usa argumentos posicionais para evitar erro de parâmetros renomeados
            result = client.predict(
                prompt,
                15.0,
                steps,
                api_name="/text-to-3d",
            )

            model_path = result if isinstance(result, str) else result[0]
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            model_hash = compute_hash(model_bytes)
            duration = round(time.time() - t0)

            return JSONResponse({
                "success": True,
                "modelUrl": f"data:model/gltf-binary;base64,{__import__('base64').b64encode(model_bytes).decode()}",
                "format": "glb",
                "hash": model_hash,
                "ipfsCid": f"Qm{model_hash[:44]}",
                "engine": "shap-e",
                "duration": duration,
                "mock": False,
            })

        else:
            raise HTTPException(status_code=400, detail="Parâmetros inválidos")

    except Exception as e:
        fake_data = f"{mode}{prompt}{time.time()}".encode()
        hash_val = compute_hash(fake_data)
        return JSONResponse({
            "success": True,
            "modelUrl": "",
            "format": "stl",
            "hash": hash_val,
            "ipfsCid": f"Qm{hash_val[:44]}",
            "engine": "fallback",
            "duration": round(time.time() - t0),
            "mock": True,
            "engineError": str(e),
        })


@app.post("/hash")
async def hash_data(
    data: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    if file:
        content = await file.read()
        return {"hash": compute_hash(content), "source": "file", "filename": file.filename}
    if data:
        return {"hash": compute_hash(data.encode()), "source": "text"}
    raise HTTPException(status_code=400, detail="Nenhum dado enviado")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
