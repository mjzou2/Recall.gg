## Run backend

cd .../vodcomms/backend

source .venv/bin/activate

python -m uvicorn main:app --reload --port 8000

## Run frontend

cd .../vodcomms/frontend

npm run dev
