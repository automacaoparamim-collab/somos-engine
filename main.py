"""
SOMOS Engine v3.1 — FastAPI Backend
Auto-descobre endpoints dos Spaces via view_api()

Imagem → 3D: Hunyuan3D-2.1 → Hunyuan3D-2 → TRELLIS.2 → Shap-E
Texto  → 3D: Hunyuan3D-2   → Shap-E
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

QUALITY_STEPS = {"low": 12, "standard": 20, "ultra": 30}

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

def encode_model(model_bytes: bytes) -> str:
    return base64.b64encode(model_bytes).decode()

def extract_path(result):
    """Extrai caminho de arquivo de qualquer formato de retorno Gradio"""
    if isinstance(result, str) and (result.endswith('.glb') or result.endswith('.obj') or result.endswith('.stl') or os.path.exists(result)):
        return result
    if isinstance(result, (list, tuple)) and len(result) > 0:
        for item in result:
            if isinstance(item, str) and os.path.exists(item):
                return item
            if isinstance(item, dict):
                for key in ("name", "path", "url", "value"):
                    val = item.get(key)
                    if val and isinstance(val, str):
                        return val
            if isinstance(item, (list, tuple)) and len(item) > 0:
                sub = item[0]
                if isinstance(sub, str):
                    return sub
                if isinstance(sub, dict):
                    for key in ("name", "path", "url"):
                        val = sub.get(key)
                        if val:
                            return val
    if isinstance(result, dict):
        for key in ("name", "path", "url", "value"):
            val = result.get(key)
            if val and isinstance(val, str):
                return val
    # tenta converter qualquer string como último recurso
    if isinstance(result, str):
        return result
    raise ValueError(f"Não foi possível extrair caminho: {str(result)[:300]}")

def mock_response(mode: str, prompt: str, t0: float, error: str):
    fake_data = f"{mode}{prompt}{time.time()}".encode()
    h = compute_hash(fake_data)
    return JSONResponse({
        "success": True, "modelUrl": "", "format": "stl",
        "hash": h, "ipfsCid": f"Qm{h[:44]}",
        "engine": "fallback", "duration": round(time.time() - t0),
        "mock": True, "engineError": error,
    })

def get_space_endpoints(space: str) -> list:
    """Retorna lista de endpoints disponíveis no Space"""
    try:
        client = Client(space, token=HF_TOKEN)
        info = client.view_api(all_endpoints=True, print_info=False)
        return list(info.keys()) if isinstance(info, dict) else []
    except Exception as e:
        return []

# ─────────────────────────────────────────────
# ENGINES — IMAGEM → 3D
# ─────────────────────────────────────────────

def try_hunyuan_image(tmp_path: str, steps: int) -> tuple:
    """Tenta Hunyuan3D-2.1 e 2.0 com descoberta automática de endpoint"""
    for space in ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2"]:
        try:
            client = Client(space, token=HF_TOKEN)
            endpoints = get_space_endpoints(space)

            # Descobre o endpoint correto
            img_endpoint = None
            for ep in ["/generate", "/image_to_3d", "/predict", "/run"]:
                if ep in endpoints:
                    img_endpoint = ep
                    break
            # Se não achou nenhum conhecido, usa o primeiro disponível
            if not img_endpoint and endpoints:
                img_endpoint = endpoints[0]
            if not img_endpoint:
                continue

            result = client.predict(
                handle_file(tmp_path),
                api_name=img_endpoint,
            )
            return extract_path(result), space.split("/")[1]
        except Exception as e:
            last_error = str(e)
            continue
    raise ValueError(f"Hunyuan falhou: {last_error}")


def try_trellis_image(tmp_path: str, steps: int) -> tuple:
    """Tenta TRELLIS.2 e community com parâmetros corretos"""
    spaces = [
        ("microsoft/TRELLIS.2",        "/image_to_3d"),
        ("trellis-community/TRELLIS",   "/image_to_3d"),
        ("JeffreyXiang/TRELLIS",        "/image_to_3d"),
    ]
    for space, endpoint in spaces:
        try:
            client = Client(space, token=HF_TOKEN)

            # TRELLIS usa resolutions como string choices: '512','1024','1536'
            result = client.predict(
                handle_file(tmp_path),  # imagem
                "1024",                 # output_format (resolução)
                0,                      # seed
                api_name=endpoint,
            )
            return extract_path(result), space.split("/")[1]
        except Exception as e:
            # tenta sem o parâmetro de resolução
            try:
                result = client.predict(
                    handle_file(tmp_path),
                    api_name=endpoint,
                )
                return extract_path(result), space.split("/")[1]
            except Exception:
                continue
    raise ValueError("TRELLIS indisponível")


def try_shapeE_image(tmp_path: str, steps: int) -> tuple:
    """Shap-E image-to-3D como último fallback"""
    client = Client("hysts/Shap-E", token=HF_TOKEN)
    result = client.predict(
        handle_file(tmp_path),
        api_name="/image-to-3d",
    )
    return extract_path(result), "shap-e-img"


# ─────────────────────────────────────────────
# ENGINES — TEXTO → 3D
# ─────────────────────────────────────────────

def try_hunyuan_text(prompt: str, steps: int) -> tuple:
    """Hunyuan3D-2 text-to-3D com descoberta de endpoint"""
    space = "tencent/Hunyuan3D-2"
    try:
        client = Client(space, token=HF_TOKEN)
        endpoints = get_space_endpoints(space)

        txt_endpoint = None
        for ep in ["/text_to_3d", "/generate", "/predict"]:
            if ep in endpoints:
                txt_endpoint = ep
                break
        if not txt_endpoint and endpoints:
            txt_endpoint = endpoints[0]
        if not txt_endpoint:
            raise ValueError("Nenhum endpoint encontrado")

        result = client.predict(prompt, api_name=txt_endpoint)
        return extract_path(result), "hunyuan3d-2"
    except Exception as e:
        raise ValueError(f"Hunyuan text falhou: {e}")


def try_shapeE_text(prompt: str, steps: int) -> tuple:
    """Shap-E text-to-3D — estável e confiável"""
    client = Client("hysts/Shap-E", token=HF_TOKEN)
    result = client.predict(prompt, 15.0, steps, api_name="/text-to-3d")
    return extract_path(result), "shap-e"


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


@app.get("/status")
async def status():
    spaces = [
        "tencent/Hunyuan3D-2.1",
        "tencent/Hunyuan3D-2",
        "microsoft/TRELLIS.2",
        "hysts/Shap-E",
    ]
    results = {}
    for space in spaces:
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"https://huggingface.co/api/spaces/{space}",
                    headers=headers,
                )
            data = r.json()
            results[space] = {
                "status": data.get("runtime", {}).get("stage", "unknown"),
                "ok": r.status_code == 200,
            }
        except Exception as e:
            results[space] = {"status": "error", "error": str(e)[:80]}

    return {"api": "online", "hf_token": "configured" if HF_TOKEN else "not set", "spaces": results}


@app.get("/api_info/{owner}/{repo}")
async def api_info(owner: str, repo: str):
    """Inspeciona endpoints disponíveis de um Space — útil para debug"""
    space = f"{owner}/{repo}"
    try:
        client = Client(space, token=HF_TOKEN)
        info = client.view_api(all_endpoints=True, print_info=False)
        return {"space": space, "endpoints": info}
    except Exception as e:
        raise HTTPException(500, f"Erro ao inspecionar {space}: {e}")


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

    model_path  = None
    engine_used = None

    try:
        # ══════════════════════════════════════
        # IMAGEM → 3D
        # ══════════════════════════════════════
        if mode in ("image", "camera"):
            if not image:
                return mock_response(mode, prompt, t0, "Imagem não recebida")

            image_bytes = await image.read()
            tmp_path = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            # Cadeia: Hunyuan → TRELLIS → ShapE-img
            for fn in [
                lambda: try_hunyuan_image(tmp_path, steps),
                lambda: try_trellis_image(tmp_path, steps),
                lambda: try_shapeE_image(tmp_path, steps),
            ]:
                try:
                    model_path, engine_used = fn()
                    break
                except Exception as e:
                    errors.append(str(e)[:120])

        # ══════════════════════════════════════
        # TEXTO → 3D
        # ══════════════════════════════════════
        else:
            if not prompt:
                return mock_response(mode, prompt, t0, "Prompt vazio")

            for fn in [
                lambda: try_hunyuan_text(prompt, steps),
                lambda: try_shapeE_text(prompt, steps),
            ]:
                try:
                    model_path, engine_used = fn()
                    break
                except Exception as e:
                    errors.append(str(e)[:120])

        if not model_path:
            return mock_response(mode, prompt, t0, " | ".join(errors))

        # ── Lê arquivo gerado ─────────────────
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
