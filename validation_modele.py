import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("✔️  VALIDATION DU MODÈLE BAYÉSIEN (sur données de TEST)")
print("="*70)

# Charger les données de TEST (20% non vues par le modèle)
df = pd.read_csv('ma_base_donnees_test.csv')
print(f"\n📊 Données de TEST : {len(df)} étudiants (20% du dataset)")
print(f"   ⚠️  Ces données n'ont PAS été utilisées pour l'entraînement")

# Charger le modèle
with open('model_probabilities.json', 'r') as f:
    model = json.load(f)

# ============================================================================
# 1. PRÉDICTIONS DU MODÈLE
# ============================================================================
print("\n" + "="*70)
print("🔮 GÉNÉRATION DES PRÉDICTIONS")
print("="*70)

y_true = []
y_pred = []
y_proba = []

for idx, row in df.iterrows():
    # Vraie valeur
    y_true.append(row['Succes'])
    
    # Prédiction du modèle
    code = f"{row['Revision']}{row['Confiance']}{row['Malade']}"
    prob_reussite = model['all_predictions'][code]['reussite'] / 100
    y_proba.append(prob_reussite)
    
    # Prédiction binaire (seuil = 50%)
    y_pred.append(1 if prob_reussite >= 0.5 else 0)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_proba = np.array(y_proba)

print(f"✅ {len(y_pred)} prédictions générées")

# ============================================================================
# 2. MÉTRIQUES DE PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("📈 MÉTRIQUES DE PERFORMANCE")
print("="*70)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n🎯 Accuracy (Exactitude)  : {accuracy*100:.2f}%")
print(f"   → Le modèle prédit correctement {accuracy*100:.1f}% des cas")

print(f"\n🎯 Precision (Précision)  : {precision*100:.2f}%")
print(f"   → Quand le modèle prédit 'Réussite', c'est vrai dans {precision*100:.1f}% des cas")

print(f"\n🎯 Recall (Rappel)        : {recall*100:.2f}%")
print(f"   → Le modèle détecte {recall*100:.1f}% des étudiants qui réussissent")

print(f"\n🎯 F1-Score              : {f1*100:.2f}%")
print(f"   → Moyenne harmonique (équilibre précision/rappel)")

# ============================================================================
# 3. MATRICE DE CONFUSION
# ============================================================================
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*70)
print("📊 MATRICE DE CONFUSION")
print("="*70)
print(f"\n                    Prédiction")
print(f"                 Échec    Réussite")
print(f"Réel  Échec      {cm[0,0]:>4}      {cm[0,1]:>4}     (Vrais Négatifs / Faux Positifs)")
print(f"      Réussite   {cm[1,0]:>4}      {cm[1,1]:>4}     (Faux Négatifs / Vrais Positifs)")

vn, fp = cm[0, 0], cm[0, 1]
fn, vp = cm[1, 0], cm[1, 1]

print(f"\n📌 Interprétation :")
print(f"   • Vrais Positifs  (VP) : {vp} - Réussite correctement prédite")
print(f"   • Vrais Négatifs  (VN) : {vn} - Échec correctement prédit")
print(f"   • Faux Positifs   (FP) : {fp} - Prédit réussite mais échoue")
print(f"   • Faux Négatifs   (FN) : {fn} - Prédit échec mais réussit")

# ============================================================================
# 4. ANALYSE PAR SEUIL
# ============================================================================
print("\n" + "="*70)
print("🎚️  ANALYSE PAR SEUIL DE DÉCISION")
print("="*70)

seuils = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nSeuil | Accuracy | Précision | Rappel  | F1-Score")
print("-" * 55)

for seuil in seuils:
    y_pred_seuil = (y_proba >= seuil).astype(int)
    acc = accuracy_score(y_true, y_pred_seuil)
    prec = precision_score(y_true, y_pred_seuil, zero_division=0)
    rec = recall_score(y_true, y_pred_seuil)
    f1_s = f1_score(y_true, y_pred_seuil)
    print(f"{seuil:.1f}   | {acc*100:>6.1f}%  | {prec*100:>7.1f}%  | {rec*100:>6.1f}% | {f1_s*100:>6.1f}%")

# ============================================================================
# 5. VISUALISATIONS
# ============================================================================
print("\n" + "="*70)
print("📊 GÉNÉRATION DES GRAPHIQUES...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Matrice de confusion
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=['Échec', 'Réussite'], 
            yticklabels=['Échec', 'Réussite'],
            cbar_kws={'label': 'Nombre'})
ax1.set_title('📊 Matrice de Confusion', fontsize=14, fontweight='bold')
ax1.set_ylabel('Valeur Réelle', fontweight='bold')
ax1.set_xlabel('Prédiction', fontweight='bold')

# 2. Métriques
ax2 = axes[0, 1]
metriques = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
valeurs = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#667eea', '#764ba2', '#11998e', '#38ef7d']
bars = ax2.barh(metriques, valeurs, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xlim(0, 100)
ax2.set_xlabel('Score (%)', fontweight='bold')
ax2.set_title('📈 Métriques de Performance', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, valeurs):
    ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
            va='center', fontweight='bold')

# 3. Distribution des probabilités
ax3 = axes[1, 0]
ax3.hist(y_proba[y_true==0], bins=20, alpha=0.5, label='Échec (réel)', color='red', edgecolor='black')
ax3.hist(y_proba[y_true==1], bins=20, alpha=0.5, label='Réussite (réel)', color='green', edgecolor='black')
ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Seuil décision')
ax3.set_xlabel('Probabilité prédite', fontweight='bold')
ax3.set_ylabel('Nombre d\'étudiants', fontweight='bold')
ax3.set_title('📊 Distribution des Probabilités', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Courbe de calibration
ax4 = axes[1, 1]
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
true_probs = []
pred_probs = []

for i in range(len(bins)-1):
    mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
    if mask.sum() > 0:
        true_probs.append(y_true[mask].mean())
        pred_probs.append(y_proba[mask].mean())

ax4.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite', linewidth=2)
if true_probs:
    ax4.scatter(pred_probs, true_probs, s=100, color='#667eea', edgecolor='black', linewidth=2, label='Modèle', zorder=5)
    ax4.plot(pred_probs, true_probs, color='#667eea', linewidth=2, alpha=0.5)
ax4.set_xlabel('Probabilité prédite', fontweight='bold')
ax4.set_ylabel('Fréquence observée', fontweight='bold')
ax4.set_title('📈 Courbe de Calibration', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('validation_modele.png', dpi=300, bbox_inches='tight')
print("   ✅ Graphique sauvegardé : validation_modele.png")

# ============================================================================
# 6. SAUVEGARDER LES RÉSULTATS
# ============================================================================
resultats = {
    'metriques': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'matrice_confusion': {
        'vrai_positif': int(vp),
        'vrai_negatif': int(vn),
        'faux_positif': int(fp),
        'faux_negatif': int(fn)
    },
    'interpretation': {
        'accuracy_pct': f"{accuracy*100:.2f}%",
        'precision_pct': f"{precision*100:.2f}%",
        'recall_pct': f"{recall*100:.2f}%",
        'f1_pct': f"{f1*100:.2f}%"
    }
}

with open('validation_resultats.json', 'w', encoding='utf-8') as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)

print("   ✅ Résultats sauvegardés : validation_resultats.json")

# ============================================================================
# 7. VERDICT FINAL
# ============================================================================
print("\n" + "="*70)
print("🏆 VERDICT FINAL")
print("="*70)

if accuracy >= 0.90:
    verdict = "EXCELLENT"
    emoji = "🌟"
elif accuracy >= 0.80:
    verdict = "TRÈS BON"
    emoji = "✅"
elif accuracy >= 0.70:
    verdict = "BON"
    emoji = "👍"
else:
    verdict = "À AMÉLIORER"
    emoji = "⚠️"

print(f"\n{emoji} Performance du modèle : {verdict}")
print(f"   Accuracy : {accuracy*100:.2f}%")
print(f"\n💡 Le modèle bayésien prédit correctement la réussite")
print(f"   dans {accuracy*100:.1f}% des cas sur {len(df)} étudiants.")

print("\n" + "="*70)