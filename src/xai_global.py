import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xai_commun import Xte, m

Xs = Xte.sample(min(400, len(Xte)), random_state=0)
raw = shap.TreeExplainer(m)(Xs)
if hasattr(raw, 'values'):
    values = raw.values
    base = raw.base_values
else:
    values = raw
    base = None
if isinstance(values, list):
    values = values[1]
if values.ndim == 3:
    values = values[:, :, 1]
    if base is not None and np.ndim(base) > 1:
        base = base[:, 1]
if base is None:
    base_value = float(np.mean(m.predict_proba(Xs)[:, 1]))
else:
    base_value = float(np.asarray(base).reshape(-1)[0])
print('valeur de base :', round(base_value, 4))
importance = pd.Series(np.abs(values).mean(0), index=Xs.columns)
print(importance.sort_values(ascending=False).round(4).to_string())
importance.round(4).to_json('reports/shap_importance.json')

display_values = shap.Explanation(values=values, base_values=np.repeat(base_value, len(Xs)), data=Xs.values, feature_names=list(Xs.columns))
shap.plots.beeswarm(display_values, show=False)
plt.tight_layout()
plt.savefig('reports/shap_global.png', dpi=120)
plt.close('all')
plt.scatter(Xs['montant'], values[:, list(Xs.columns).index('montant')], c=Xs['nuit'], cmap='coolwarm', s=12)
plt.xscale('log')
plt.xlabel('montant (MAD)')
plt.ylabel('contribution SHAP du montant')
plt.colorbar(label='nuit')
plt.tight_layout()
plt.savefig('reports/shap_montant.png', dpi=120)
print('graphiques ecrits')
