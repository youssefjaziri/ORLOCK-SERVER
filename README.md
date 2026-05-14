# ORLOCK-SERVER

## Launch The Project

### 1) Local run
```bash
# from repository root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run API server
uvicorn src.orlock.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Docker run
```bash
# build image
docker build -t orlock-server .

# run container
docker run --rm -it -p 8000:8000 orlock-server
```

## Notes
- Main app entrypoint is `src/orlock/main.py`.
- Optional transcription outputs are written under `transcriptions/`.
