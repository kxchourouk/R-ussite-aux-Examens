import pandas as pd
import numpy as np

print("🎲 Génération de données simulées basées sur le modèle théorique...")

n_students = 1000
np.random.seed(42)

data = []

for i in range(n_students):
    # Variables observables AVANT l'examen
    revision = np.random.choice([0, 1], p=[0.35, 0.65])
    confiance = np.random.choice([0, 1], p=[0.45, 0.55])
    malade = np.random.choice([0, 1], p=[0.80, 0.20])
    
    # Calcul probabilité de succès
    p_success = 0.70
    
    if revision == 1:
        p_success *= 1.3
    else:
        p_success *= 0.6
    
    if confiance == 1:
        p_success *= 1.2
    else:
        p_success *= 0.7
    
    if malade == 1:
        p_success *= 0.5
    else:
        p_success *= 1.1
    
    p_success = min(max(p_success, 0.05), 0.95)
    succes = np.random.choice([0, 1], p=[1-p_success, p_success])
    
    data.append({
        'Revision': revision,
        'Confiance': confiance,
        'Malade': malade,
        'Succes': succes
    })

df = pd.DataFrame(data)

# Statistiques globales
print(f"\n📊 Statistiques globales :")
print(f"   - Total : {len(df)} étudiants")
print(f"   - Réussite : {df['Succes'].mean()*100:.1f}%")
print(f"   - Révision : {df['Revision'].mean()*100:.1f}%")
print(f"   - Confiance : {df['Confiance'].mean()*100:.1f}%")
print(f"   - Maladie : {df['Malade'].mean()*100:.1f}%")

# ============================================================================
# SPLIT TRAIN/TEST : 80% / 20%
# ============================================================================
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Succes'])

print(f"\n📊 Split Train/Test :")
print(f"   - Données d'entraînement : {len(df_train)} étudiants (80%)")
print(f"   - Données de validation  : {len(df_test)} étudiants (20%)")

# Vérifier la distribution
print(f"\n📊 Distribution Train :")
print(f"   - Réussite : {df_train['Succes'].mean()*100:.1f}%")
print(f"\n📊 Distribution Test :")
print(f"   - Réussite : {df_test['Succes'].mean()*100:.1f}%")

# Analyse par combinaison (sur les données complètes pour info)
print(f"\n🔍 Analyse des combinaisons (données complètes) :")
for rev in [0, 1]:
    for conf in [0, 1]:
        for mal in [0, 1]:
            subset = df[(df['Revision']==rev) & (df['Confiance']==conf) & (df['Malade']==mal)]
            if len(subset) > 0:
                rate = subset['Succes'].mean() * 100
                print(f"   R={rev}, C={conf}, M={mal} → {rate:.1f}% (n={len(subset)})")

# Sauvegarder les 3 fichiers
df.to_csv('ma_base_donnees.csv', index=False)  # Toutes les données (pour compatibilité)
df_train.to_csv('ma_base_donnees_train.csv', index=False)
df_test.to_csv('ma_base_donnees_test.csv', index=False)

print(f"\n✅ Fichiers créés :")
print(f"   - 'ma_base_donnees.csv' (toutes les données)")
print(f"   - 'ma_base_donnees_train.csv' (80% - entraînement)")
print(f"   - 'ma_base_donnees_test.csv' (20% - validation)")