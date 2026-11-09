# TP07 - Explicabilite et auditabilite

Le registre TP2 n'etant pas present dans cet espace, ce projet utilise la reconstruction de l'Annexe A.

## Demarrage PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force data, reports | Out-Null
python donnees.py
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Dans un second terminal active, executer :

```powershell
python src/reconstruire_fraude.py
python src/verifier_modele.py
python src/xai_importance.py
python src/xai_global.py
python src/xai_local.py
python src/xai_lime.py
python src/interpretables.py
python src/journaliser.py
python src/publier_card.py
```

Les sorties sont dans `reports/`. Le modele est charge par l'alias MLflow `fraude@production`.
