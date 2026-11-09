import mlflow
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from xai_commun import CAS, VERSION, Xte, m, yte

run = mlflow.get_run(VERSION.run_id)
pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, 1]
print('modele     : fraude, version', VERSION.version, '| run', VERSION.run_id[:8])
print('classe     :', type(m).__name__, '|', {k: run.data.params[k] for k in ['n_estimators', 'max_depth']})
print('F1         :', round(f1_score(yte, pred), 4), '| F1 enregistre :', round(run.data.metrics['f1'], 4))
print('precision  :', round(precision_score(yte, pred), 4), '| rappel :', round(recall_score(yte, pred), 4), '| AUC :', round(roc_auc_score(yte, proba), 4))
print('matrice    :', confusion_matrix(yte, pred).tolist())
print('Mme Benali : P(fraude) =', round(m.predict_proba(CAS)[0, 1], 4))
