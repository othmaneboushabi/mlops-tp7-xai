import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xai_commun import CAS, X, Xte, Xtr, m, yte, ytr

results = {}


def note(name, model, features):
    predictions = model.predict(features)
    results[name] = {'f1': round(f1_score(yte, predictions), 4), 'fraudes_detectees': int(((predictions == 1) & (yte == 1)).sum())}
    print(f'{name:28s} F1 = {results[name]["f1"]:.4f} | fraudes detectees : {results[name]["fraudes_detectees"]} / {int(yte.sum())}')


def log_amount(data):
    transformed = data.copy()
    transformed['log_montant'] = np.log10(transformed.pop('montant'))
    return transformed


note('foret fraude@production', m, Xte)
lr_raw = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
note('logistique, montant brut', lr_raw, Xte)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(log_amount(Xtr), ytr)
note('logistique, log10(montant)', lr, log_amount(Xte))
print('coefficients standardises :', {column: round(float(value), 2) for column, value in zip(log_amount(Xtr).columns, lr[-1].coef_[0])})

for depth in [2, 3, 6, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    note(f'arbre profondeur {depth} ({tree.get_n_leaves()} feuilles)', tree, Xte)
tree3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
print(export_text(tree3, feature_names=list(X.columns), decimals=1))
print('Mme Benali : feuille/nœud', tree3.apply(CAS)[0], '| P =', round(tree3.predict_proba(CAS)[0, 1], 3))
json.dump(results, open('reports/interpretables.json', 'w'), indent=2)
