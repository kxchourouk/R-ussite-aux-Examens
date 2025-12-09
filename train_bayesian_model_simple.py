import pandas as pd
import numpy as np
import joblib
import json
from collections import defaultdict

print("📊 Chargement des données d'entraînement...")
df = pd.read_csv('ma_base_donnees_train.csv')
print(f"✅ Données d'entraînement chargées : {len(df)} étudiants (80% du dataset)")
print(f"   (Le modèle sera validé sur les 20% restants)")

# 1. Calculer les probabilités directement (sans pgmpy)
print("\n📈 Calcul des probabilités bayésiennes...")

# Probabilités a priori
total = len(df)
success_count = df['Succes'].sum()
failure_count = total - success_count

prior_success = success_count / total
prior_failure = failure_count / total

# 2. Calculer les tables de probabilités conditionnelles
def calculate_conditional_prob(df, condition_col, target_col):
    """Calcule P(target=1|condition) et P(target=0|condition)"""
    probs = {}
    
    # Pour chaque valeur de la condition (0 ou 1)
    for cond_val in [0, 1]:
        subset = df[df[condition_col] == cond_val]
        if len(subset) > 0:
            # P(Succes=1 | condition=cond_val)
            p_success = subset[target_col].mean()
            probs[f'cond_{cond_val}_success'] = float(p_success)
            probs[f'cond_{cond_val}_failure'] = float(1 - p_success)
        else:
            probs[f'cond_{cond_val}_success'] = 0.0
            probs[f'cond_{cond_val}_failure'] = 0.0
    
    return probs

# 3. Calculer toutes les probabilités nécessaires
print("🔍 Calcul des distributions...")

# Pour Révision
rev_probs = calculate_conditional_prob(df, 'Succes', 'Revision')
conf_probs = calculate_conditional_prob(df, 'Succes', 'Confiance')
mal_probs = calculate_conditional_prob(df, 'Succes', 'Malade')

# 4. Fonction de prédiction bayésienne
def bayesian_predict(revision, confiance, malade):
    """Calcule P(Succes=1 | Revision, Confiance, Malade)"""
    
    # P(Succes=1) * P(observations | Succes=1)
    p_success_given_obs = prior_success
    p_success_given_obs *= rev_probs['cond_1_success'] if revision == 1 else rev_probs['cond_1_failure']
    p_success_given_obs *= conf_probs['cond_1_success'] if confiance == 1 else conf_probs['cond_1_failure']
    p_success_given_obs *= mal_probs['cond_1_success'] if malade == 1 else mal_probs['cond_1_failure']
    
    # P(Succes=0) * P(observations | Succes=0)
    p_failure_given_obs = prior_failure
    p_failure_given_obs *= rev_probs['cond_0_success'] if revision == 1 else rev_probs['cond_0_failure']
    p_failure_given_obs *= conf_probs['cond_0_success'] if confiance == 1 else conf_probs['cond_0_failure']
    p_failure_given_obs *= mal_probs['cond_0_success'] if malade == 1 else mal_probs['cond_0_failure']
    
    # Normalisation
    total_prob = p_success_given_obs + p_failure_given_obs
    
    if total_prob == 0:
        return 0.5  # Valeur par défaut si aucune donnée
    
    return p_success_given_obs / total_prob

# 5. Tester la fonction
print("\n🧪 Test des prédictions...")
test_cases = [
    (1, 1, 0),  # Révision=Oui, Confiance=Oui, Malade=Non
    (0, 0, 1),  # Révision=Non, Confiance=Non, Malade=Oui
    (1, 0, 0),  # Révision=Oui, Confiance=Non, Malade=Non
]

for rev, conf, mal in test_cases:
    prob = bayesian_predict(rev, conf, mal)
    print(f"  Revision={rev}, Confiance={conf}, Malade={mal} → {prob*100:.1f}% de réussite")

# 6. Pré-calculer toutes les combinaisons possibles (8 combinaisons)
print("\n📋 Pré-calcul de toutes les combinaisons...")
all_predictions = {}

for rev in [0, 1]:
    for conf in [0, 1]:
        for mal in [0, 1]:
            prob = bayesian_predict(rev, conf, mal)
            all_predictions[f"{rev}{conf}{mal}"] = {
                'reussite': float(prob * 100),
                'echec': float((1 - prob) * 100)
            }

# 7. Sauvegarder le modèle
model_data = {
    'prior': {
        'reussite': float(prior_success),
        'echec': float(prior_failure)
    },
    'conditional': {
        'rev_reussite': float(rev_probs['cond_1_success']),  # P(Revision=1 | Succes=1)
        'rev_echec': float(rev_probs['cond_0_success']),     # P(Revision=1 | Succes=0)
        'conf_reussite': float(conf_probs['cond_1_success']),
        'conf_echec': float(conf_probs['cond_0_success']),
        'mal_reussite': float(mal_probs['cond_1_success']),
        'mal_echec': float(mal_probs['cond_0_success'])
    },
    'all_predictions': all_predictions,
    'dataset_stats': {
        'total': int(total),
        'success_rate': f"{prior_success*100:.1f}%",
        'revision_rate': f"{df['Revision'].mean()*100:.1f}%",
        'confidence_rate': f"{df['Confiance'].mean()*100:.1f}%",
        'health_rate': f"{(1-df['Malade'].mean())*100:.1f}%",
        'success_count': int(success_count),
        'failure_count': int(failure_count)
    }
}

# Sauvegarder
joblib.dump(model_data, 'bayesian_model_simple.joblib')

# Exporter pour JavaScript
with open('model_probabilities.json', 'w') as f:
    json.dump(model_data, f, indent=2)

print("\n✅ Modèle bayésien créé avec succès !")
print("📊 Statistiques :")
print(f"   - Étudiants : {total}")
print(f"   - Réussite : {prior_success*100:.1f}% ({success_count} étudiants)")
print(f"   - Échec : {prior_failure*100:.1f}% ({failure_count} étudiants)")
print(f"   - Taux de révision : {df['Revision'].mean()*100:.1f}%")
print(f"   - Taux de confiance : {df['Confiance'].mean()*100:.1f}%")
print(f"   - Taux de bonne santé : {(1-df['Malade'].mean())*100:.1f}%")

print("\n📁 Fichiers créés :")
print("   - bayesian_model_simple.joblib")
print("   - model_probabilities.json")