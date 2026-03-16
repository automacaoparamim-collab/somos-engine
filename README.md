# SOMOS Engine — FastAPI Backend

Deploy no Railway.app — sem limite de tempo por request.

## Deploy (5 minutos)

### 1. Criar repositório GitHub separado

```bash
cd somos-engine
git init
git add .
git commit -m "init somos engine"
git branch -M main
# Crie repo no GitHub: somos-engine
git remote add origin https://SEU_TOKEN@github.com/automacaoparamim-collab/somos-engine.git
git push -u origin main
```

### 2. Deploy no Railway

1. Acesse [railway.app](https://railway.app)
2. **Login with GitHub**
3. **New Project → Deploy from GitHub repo**
4. Selecione `somos-engine`
5. Railway detecta Python automaticamente e faz deploy

### 3. Adicionar variável de ambiente

No Railway → seu projeto → **Variables**:
```
HF_TOKEN = seu_token_aqui
```

### 4. Pegar a URL pública

Railway → seu projeto → **Settings → Domains → Generate Domain**

Vai gerar algo como: `somos-engine-production.up.railway.app`

### 5. Atualizar o Vercel

No Vercel → Environment Variables, adicione:
```
NEXT_PUBLIC_ENGINE_URL = https://somos-engine-production.up.railway.app
```

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Healthcheck |
| GET | `/status` | Status HF Spaces + token |
| POST | `/generate` | Gera modelo 3D |
| POST | `/hash` | SHA-256 de texto/arquivo |

## Teste local

```bash
pip install -r requirements.txt
HF_TOKEN=hf_xxx uvicorn main:app --reload
# http://localhost:8000/status
```
