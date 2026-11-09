from pathlib import Path
import hashlib
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()
version = client.get_model_version_by_alias('fraude', 'production')
run_id = Path('reports/run_xai.txt').read_text(encoding='utf-8').strip()
data_path = Path('data/transactions.csv')
data_md5 = hashlib.md5(data_path.read_bytes()).hexdigest()
card = Path('MODEL_CARD.md').read_text(encoding='utf-8')
card = card.replace('renseigne automatiquement apres reconstruction', f'{version.run_id}')
card = card.replace('part de fraudes, empreinte et origine : renseignees automatiquement', f'empreinte MD5 : {data_md5}')
with mlflow.start_run(run_id=run_id):
    mlflow.log_text(card, 'MODEL_CARD.md')
client.update_model_version('fraude', version.version, description=card)
client.set_model_version_tag('fraude', version.version, 'model_card', f'MODEL_CARD.md, run {run_id[:8]}')
Path('MODEL_CARD.md').write_text(card, encoding='utf-8')
print('Model Card attachee a fraude v' + version.version + ' et au run', run_id[:8])
