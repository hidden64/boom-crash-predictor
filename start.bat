@echo off
echo ====================================================
echo   Lancement de la Tour de Controle Boom-Crash IA
echo ====================================================

echo [1/2] Demarrage du Backend (FastAPI)...
start cmd /k "cd backend && title Backend FastAPI && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Demarrage du Frontend (Next.js)...
start cmd /k "cd frontend && title Frontend Next.js && npm run dev"

echo Succes! Le backend tourne sur le port 8000.
echo Le frontend va s'ouvrir sur http://localhost:3000.
pause
