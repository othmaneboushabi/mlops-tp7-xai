import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
from xai_commun import CAS, Xte, m

explainer = shap.TreeExplainer(m)
proba = m.predict_proba(Xte)[:, 1]
idx = Xte.index[proba.argmax()]
for name, row in [('benali', CAS), (f'test_{idx}', Xte.loc[[idx]])]:
    raw = explainer(row)
    values = raw.values
    base = raw.base_values
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1]
    base_value = float(np.asarray(base).reshape(-1)[0])
    explanation = shap.Explanation(values=values[0], base_values=base_value, data=row.iloc[0].values, feature_names=list(row.columns))
    print(f'--- {name} :', row.iloc[0].to_dict())
    print('valeur de base :', round(base_value, 4))
    for column, contribution in sorted(zip(row.columns, explanation.values), key=lambda item: -abs(item[1])):
        print(f' {column:11s} {contribution:+.4f}')
    prediction = m.predict_proba(row)[0, 1]
    print('base + somme   :', round(base_value + explanation.values.sum(), 4), '| prediction :', round(prediction, 4))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/shap_{name}.png', dpi=120)
    plt.close('all')
