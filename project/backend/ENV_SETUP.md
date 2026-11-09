# Backend .env setup — Tabular Q

The backend automatically reads `project/backend/.env` before creating
services. The actual `.env` is ignored by Git. Only `.env.example` is tracked.

From repository root on Windows PowerShell, after checking out this branch:

```powershell
cd project/backend
Copy-Item .env.example .env
python -m pip install -e ".[api]"
fastapi dev src/api/main.py
```

The `.env` contains:

```dotenv
XIANGQI_TABULAR_Q_CHECKPOINT=models/tabular_q/q_table.json.gz
```

Place your downloaded `q_table.json.gz` in
`project/backend/models/tabular_q/q_table.json.gz`. Do not extract it.
The relative checkpoint path is always resolved against the backend root,
independent of the terminal working directory.

## Previously exported PowerShell variables

OS environment variables take precedence over `.env`. If you previously
set a different `XIANGQI_TABULAR_Q_CHECKPOINT` value in PowerShell, stop
FastAPI with Ctrl+C and run the following in the **same terminal** before
restarting:

```powershell
Remove-Item Env:XIANGQI_TABULAR_Q_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:XIANGQI_TABULAR_Q_DRIVE_FILE_ID -ErrorAction SilentlyContinue
fastapi dev src/api/main.py
```

This does not erase the new `.env`. Local file mode does not require
Google authentication. Verify readiness at
`http://127.0.0.1:8000/api/v1/models/tabular-q/status`; then refresh
the frontend for the enabled Tabular Q button.

## Tests

```powershell
python -m pip install -e ".[api,test]"
python -m unittest tests.api.test_dotenv_config -v
python -m unittest tests.api.test_tabular_q_status_api -v
```
