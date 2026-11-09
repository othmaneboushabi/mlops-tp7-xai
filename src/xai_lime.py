from lime.lime_tabular import LimeTabularExplainer
from xai_commun import CAS, X, Xtr, m


def expliquer(graine, nombre):
    explainer = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns), class_names=['normal', 'fraude'], mode='classification', random_state=graine)
    return explainer.explain_instance(CAS.values[0], m.predict_proba, num_features=4, num_samples=nombre)


explanation = expliquer(0, 2000)
print('regles :', [(rule, round(weight, 3)) for rule, weight in explanation.as_list()])
explanation.save_to_file('reports/lime_benali.html')
figure = explanation.as_pyplot_figure()
figure.tight_layout()
figure.savefig('reports/lime_benali.png', dpi=120)

for nombre in [500, 2000]:
    print(f'--- {nombre} voisins     montant    nuit anciennete freq_24h')
    for graine in range(5):
        weights = dict(expliquer(graine, nombre).as_list())
        values = [next(value for rule, value in weights.items() if column in rule) for column in X.columns]
        print(f'graine {graine}       {values[0]:+.3f} {values[3]:+.3f}      {values[1]:+.3f}    {values[2]:+.3f}')
