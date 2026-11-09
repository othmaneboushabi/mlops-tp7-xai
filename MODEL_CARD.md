# Model Card - fraude (version servie par l'alias production)

## Modele
Nom et version dans le registre MLflow : fraude, version 1, alias production.
Famille et reglages : RandomForestClassifier, 500 arbres, profondeur maximale 16, random_state 42.
Run d'origine (MLflow) : 913401778b9640d99f6545c240724088.

## Objectif et usage prevu
Detecter les paiements susceptibles d'etre frauduleux afin de declencher une verification humaine.

## Usages exclus
Ne pas bloquer definitivement un compte ou sanctionner une personne sans revue humaine.

## Donnees
Fichier : data/transactions.csv, 5 000 lignes ; empreinte MD5 : cf9f6f86c141e7df32c61715c493483f.
Origine des donnees : synthetiques, generees avec la graine 7 selon l'Annexe A du TP07.

## Performances (jeu de test, 1 250 transactions)
F1, precision, rappel et AUC : renseignes automatiquement apres verification.

## Methodes d'explicabilite
Globale : permutation et SHAP TreeExplainer sur 400 transactions de test.
Locale : SHAP waterfall et LIME sur le cas de Mme Benali ; LIME est sensible au tirage des voisins.
Variables les plus influentes : montant, anciennete, nuit et freq_24h, selon la methode.

## Comparaison avec un modele interpretable
La logistique avec log10(montant) est presque aussi performante mais plus lisible ; l'arbre court est plus simple et moins performant.

## Limites et risques connus
Les donnees sont synthetiques et le jeu de test ne contient que 134 fraudes. SHAP et LIME expliquent le modele sans etablir de causalite.

## Recommandations d'utilisation et de surveillance
Surveiller F1, precision, rappel, taux de faux positifs, derive des montants et stabilite des explications apres chaque changement de modele ou d'alias.

## Tracabilite
Run des explications (experience tp7-explicabilite) : renseigne automatiquement.
Auteurs : groupe TP07   Date : 2026-11-09   Prochaine revue : apres chaque nouveau modele en production.
