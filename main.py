"""
SOMOS Engine v3.3
Texto → 3D:  Shap-E (endpoint correto) + Hunyuan3D-2
Imagem → 3D: Hunyuan3D-2.1 → TRELLIS.2 → Shap-E
"""

import os, hashlib, time, base64, httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from typing import Optional
import uvicorn

HF_TOKEN = os.environ.get("HF_TOKEN")

# Incrementamos steps padrão para melhor qualidade
QUALITY_STEPS = {"low": 20, "standard": 40, "ultra": 64}
QUALITY_RES   = {"low": "512", "standard": "1024", "ultra": "1536"}

app = FastAPI(title="SOMOS Engine", version="3.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── UTILS ───────────────────────────────────────────────────────────────────

def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def encode_model(model_bytes: bytes) -> str:
    return base64.b64encode(model_bytes).decode()

def extract_path(result):
    if isinstance(result, str): return result
    if isinstance(result, (list, tuple)):
        for item in result:
            p = extract_path(item)
            if p: return p
    if isinstance(result, dict):
        for key in ("name","path","url","value"):
            v = result.get(key)
            if v and isinstance(v, str): return v
    return None

def mock_response(mode, prompt, t0, error):
    h = compute_hash(f"{mode}{prompt}{time.time()}".encode())
    return JSONResponse({"success":True,"modelUrl":"","format":"stl",
        "hash":h,"ipfsCid":f"Qm{h[:44]}","engine":"fallback",
        "duration":round(time.time()-t0),"mock":True,"engineError":error})

# ─── TEXTO → 3D ──────────────────────────────────────────────────────────────

def try_shapeE_text(prompt: str, steps: int) -> tuple:
    """
    Shap-E text-to-3D.
    Usa view_api() para descobrir o endpoint correto em runtime.
    """
    client = Client("hysts/Shap-E", token=HF_TOKEN)

    # Descobre endpoints disponíveis
    try:
        info = client.view_api(all_endpoints=True, print_info=False)
        endpoints = list(info.keys()) if isinstance(info, dict) else []
    except Exception:
        endpoints = []

    errors = []
    # Prioridade de endpoints conhecidos para Shap-E
    for ep in ["/text-to-3d", "/text_to_3d", "/run", "/predict"]:
        if ep not in endpoints and endpoints:
            continue
        try:
            result = client.predict(prompt, 15.0, steps, api_name=ep)
            path = extract_path(result)
            if path: return path, "shap-e"
        except Exception as e:
            errors.append(f"{ep}: {e}")
            continue

    # Tenta sem api_name (usa o default)
    try:
        result = client.predict(prompt, 15.0, steps)
        path = extract_path(result)
        if path: return path, "shap-e"
    except Exception as e:
        errors.append(f"default: {e}")

    raise ValueError(f"Shap-E text falhou: {' | '.join(errors)}")


def try_hunyuan_text(prompt: str, steps: int) -> tuple:
    """Hunyuan3D-2 text-to-3D com descoberta de endpoint"""
    client = Client("tencent/Hunyuan3D-2", token=HF_TOKEN)
    try:
        info = client.view_api(all_endpoints=True, print_info=False)
        endpoints = list(info.keys()) if isinstance(info, dict) else []
    except Exception:
        endpoints = []

    for ep in ["/text_to_3d", "/t23d", "/generate", "/predict", "/run"]:
        if ep not in endpoints and endpoints:
            continue
        try:
            result = client.predict(prompt, "", 42, steps, 7.5, api_name=ep)
            path = extract_path(result)
            if path: return path, "hunyuan3d-2"
        except Exception:
            try:
                result = client.predict(prompt, api_name=ep)
                path = extract_path(result)
                if path: return path, "hunyuan3d-2"
            except Exception:
                continue

    raise ValueError("Hunyuan text: nenhum endpoint funcionou")

# ─── IMAGEM → 3D ─────────────────────────────────────────────────────────────

def try_hunyuan_image(tmp_path: str, steps: int) -> tuple:
    """Hunyuan3D-2.1 image-to-3D — melhor qualidade"""
    for space in ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2"]:
        client = Client(space, token=HF_TOKEN)
        try:
            info = client.view_api(all_endpoints=True, print_info=False)
            endpoints = list(info.keys()) if isinstance(info, dict) else []
        except Exception:
            endpoints = []

        for ep in ["/image_to_3d", "/i23d", "/generate", "/predict", "/run"]:
            if ep not in endpoints and endpoints:
                continue
            # Tenta com parâmetros completos
            for args in [
                [handle_file(tmp_path), 42, steps, 7.5, True],
                [handle_file(tmp_path), 42, steps, 7.5],
                [handle_file(tmp_path)],
            ]:
                try:
                    result = client.predict(*args, api_name=ep)
                    path = extract_path(result)
                    if path: return path, space.split("/")[1]
                except Exception:
                    continue

    raise ValueError("Hunyuan image: falhou")


def try_trellis_image(tmp_path: str, steps: int) -> tuple:
    """TRELLIS.2 image-to-3D — boa geometria"""
    for space, ep in [
        ("microsoft/TRELLIS.2",       "/image_to_3d"),
        ("trellis-community/TRELLIS", "/image_to_3d"),
        ("JeffreyXiang/TRELLIS",      "/image_to_3d"),
    ]:
        client = Client(space, token=HF_TOKEN)
        # TRELLIS.2 pipeline: preprocess → image_to_3d
        for args in [
            # Com preprocess separado
            None,
            # Direto
            [handle_file(tmp_path), "1024", 0],
            [handle_file(tmp_path), 42, steps, 7.5, steps, 3.0],
            [handle_file(tmp_path)],
        ]:
            try:
                if args is None:
                    # Tenta preprocess primeiro
                    try:
                        pre = client.predict(handle_file(tmp_path), True, api_name="/preprocess_image")
                        img = extract_path(pre) or tmp_path
                    except Exception:
                        img = tmp_path
                    result = client.predict(handle_file(img), "1024", 0, api_name=ep)
                else:
                    result = client.predict(*args, api_name=ep)
                path = extract_path(result)
                if path: return path, space.split("/")[1]
            except Exception:
                continue

    raise ValueError("TRELLIS: todos os spaces falharam")


def try_shapeE_image(tmp_path: str) -> tuple:
    """Shap-E image-to-3D — último fallback"""
    client = Client("hysts/Shap-E", token=HF_TOKEN)
    for ep in ["/image-to-3d", "/image_to_3d", "/run", "/predict"]:
        try:
            result = client.predict(handle_file(tmp_path), api_name=ep)
            path = extract_path(result)
            if path: return path, "shap-e-img"
        except Exception:
            continue
    raise ValueError("Shap-E image: falhou")

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status":"online","engine":"SOMOS Engine v3.3","hf_token":bool(HF_TOKEN)}


@app.get("/status")
async def status():
    spaces = ["tencent/Hunyuan3D-2.1","tencent/Hunyuan3D-2","microsoft/TRELLIS.2","hysts/Shap-E"]
    results = {}
    for space in spaces:
        try:
            headers = {"Authorization":f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"https://huggingface.co/api/spaces/{space}", headers=headers)
            data = r.json()
            results[space] = {"status":data.get("runtime",{}).get("stage","unknown"),"ok":r.status_code==200}
        except Exception as e:
            results[space] = {"status":"error","error":str(e)[:80]}
    return {"api":"online","hf_token":"configured" if HF_TOKEN else "not set","spaces":results}


@app.get("/api_info/{owner}/{repo}")
async def api_info(owner:str, repo:str):
    """Debug: endpoints disponíveis de um Space"""
    space = f"{owner}/{repo}"
    try:
        client = Client(space, token=HF_TOKEN)
        info = client.view_api(all_endpoints=True, print_info=False)
        return {"space":space,"endpoints":info}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate")
async def generate(
    mode:    str                  = Form(...),
    prompt:  str                  = Form(""),
    quality: str                  = Form("standard"),
    style:   str                  = Form("realistic"),
    image:   Optional[UploadFile] = File(None),
):
    t0     = time.time()
    steps  = QUALITY_STEPS.get(quality, 40)
    errors = []

    if not HF_TOKEN:
        return mock_response(mode, prompt, t0, "HF_TOKEN não configurado")

    model_path  = None
    engine_used = None

    try:
        # ── IMAGEM → 3D ──────────────────────────────────────────────
        if mode in ("image","camera"):
            if not image:
                return mock_response(mode, prompt, t0, "Imagem não recebida")

            image_bytes = await image.read()
            tmp_path = f"/tmp/input_{int(time.time())}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)

            for fn in [
                lambda: try_hunyuan_image(tmp_path, steps),
                lambda: try_trellis_image(tmp_path, steps),
                lambda: try_shapeE_image(tmp_path),
            ]:
                try:
                    model_path, engine_used = fn()
                    if model_path: break
                except Exception as e:
                    errors.append(str(e)[:150])

        # ── TEXTO → 3D ───────────────────────────────────────────────
        else:
            if not prompt:
                return mock_response(mode, prompt, t0, "Prompt vazio")

            for fn in [
                lambda: try_shapeE_text(prompt, steps),
                lambda: try_hunyuan_text(prompt, steps),
            ]:
                try:
                    model_path, engine_used = fn()
                    if model_path: break
                except Exception as e:
                    errors.append(str(e)[:150])

        if not model_path:
            return mock_response(mode, prompt, t0, " | ".join(errors))

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
            "duration": round(time.time()-t0),
            "mock":     False,
            "warnings": errors if errors else None,
        })

    except Exception as e:
        return mock_response(mode, prompt, t0, str(e))


@app.post("/hash")
async def hash_data(data:str=Form(""), file:Optional[UploadFile]=File(None)):
    if file:
        content = await file.read()
        return {"hash":compute_hash(content),"source":"file","filename":file.filename}
    if data:
        return {"hash":compute_hash(data.encode()),"source":"text"}
    raise HTTPException(400, "Nenhum dado enviado")


if __name__ == "__main__":
    port = int(os.environ.get("PORT",8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
