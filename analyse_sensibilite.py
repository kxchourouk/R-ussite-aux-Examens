import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("📊 ANALYSE DE SENSIBILITÉ - IMPACT DES FACTEURS (Données de TEST)")
print("="*70)

# Charger les données de TEST
df = pd.read_csv('ma_base_donnees_test.csv')

print(f"\n📋 Base de données de TEST : {len(df)} étudiants (20%)")
print(f"   Taux de réussite global : {df['Succes'].mean()*100:.1f}%\n")

# ============================================================================
# 1. IMPACT MARGINAL DE CHAQUE FACTEUR
# ============================================================================
print("="*70)
print("🔍 IMPACT MARGINAL (toutes choses égales par ailleurs)")
print("="*70)

# Révision
rev_oui = df[df['Revision']==1]['Succes'].mean()
rev_non = df[df['Revision']==0]['Succes'].mean()
impact_rev = rev_oui - rev_non

print(f"\n📚 RÉVISION :")
print(f"   • Avec révision    : {rev_oui*100:.1f}% de réussite")
print(f"   • Sans révision    : {rev_non*100:.1f}% de réussite")
print(f"   → IMPACT : +{impact_rev*100:.1f} points")

# Confiance
conf_oui = df[df['Confiance']==1]['Succes'].mean()
conf_non = df[df['Confiance']==0]['Succes'].mean()
impact_conf = conf_oui - conf_non

print(f"\n💪 CONFIANCE :")
print(f"   • Avec confiance   : {conf_oui*100:.1f}% de réussite")
print(f"   • Sans confiance   : {conf_non*100:.1f}% de réussite")
print(f"   → IMPACT : +{impact_conf*100:.1f} points")

# Maladie
mal_non = df[df['Malade']==0]['Succes'].mean()
mal_oui = df[df['Malade']==1]['Succes'].mean()
impact_mal = mal_non - mal_oui

print(f"\n🏥 SANTÉ :")
print(f"   • En bonne santé   : {mal_non*100:.1f}% de réussite")
print(f"   • Malade           : {mal_oui*100:.1f}% de réussite")
print(f"   → IMPACT : +{impact_mal*100:.1f} points")

# ============================================================================
# 2. CLASSEMENT PAR IMPORTANCE
# ============================================================================
print(f"\n{'='*70}")
print("🏆 CLASSEMENT DES FACTEURS PAR IMPORTANCE")
print("="*70)

impacts = [
    ('Révision', abs(impact_rev)),
    ('Confiance', abs(impact_conf)),
    ('Santé', abs(impact_mal))
]

impacts_sorted = sorted(impacts, key=lambda x: x[1], reverse=True)

for i, (facteur, impact) in enumerate(impacts_sorted, 1):
    print(f"{i}. {facteur:<12} : {impact*100:>5.1f} points d'impact")

facteur_principal = impacts_sorted[0][0]
print(f"\n🎯 FACTEUR LE PLUS IMPORTANT : {facteur_principal.upper()}")

# ============================================================================
# 3. ANALYSE COMBINÉE (INTERACTIONS)
# ============================================================================
print(f"\n{'='*70}")
print("🔗 ANALYSE DES INTERACTIONS")
print("="*70)

print("\n📊 Meilleur cas (Révision=1, Confiance=1, Malade=0) :")
best = df[(df['Revision']==1) & (df['Confiance']==1) & (df['Malade']==0)]
if len(best) > 0:
    print(f"   → {best['Succes'].mean()*100:.1f}% de réussite (n={len(best)})")

print("\n📊 Pire cas (Révision=0, Confiance=0, Malade=1) :")
worst = df[(df['Revision']==0) & (df['Confiance']==0) & (df['Malade']==1)]
if len(worst) > 0:
    print(f"   → {worst['Succes'].mean()*100:.1f}% de réussite (n={len(worst)})")

# ============================================================================
# 4. GRAPHIQUES DE VISUALISATION
# ============================================================================
print(f"\n{'='*70}")
print("📈 GÉNÉRATION DES GRAPHIQUES...")
print("="*70)

# Créer une figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Impact des facteurs
facteurs = ['Révision', 'Confiance', 'Santé']
impacts_values = [impact_rev*100, impact_conf*100, impact_mal*100]
colors = ['#667eea', '#764ba2', '#11998e']

bars = ax1.bar(facteurs, impacts_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Impact sur la réussite (points)', fontsize=12, fontweight='bold')
ax1.set_title('📊 Impact Marginal des Facteurs', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(impacts_values) * 1.2)
ax1.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for bar, val in zip(bars, impacts_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Graphique 2 : Taux de réussite par condition
conditions = ['Révision\nOui', 'Révision\nNon', 'Confiance\nOui', 'Confiance\nNon', 
              'Santé\nBonne', 'Santé\nMalade']
taux = [rev_oui*100, rev_non*100, conf_oui*100, conf_non*100, mal_non*100, mal_oui*100]
colors2 = ['#38ef7d', '#f45c43', '#38ef7d', '#f45c43', '#38ef7d', '#f45c43']

bars2 = ax2.bar(conditions, taux, color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Taux de réussite (%)', fontsize=12, fontweight='bold')
ax2.set_title('📈 Taux de Réussite par Condition', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.axhline(y=df['Succes'].mean()*100, color='red', linestyle='--', 
           label=f'Moyenne: {df["Succes"].mean()*100:.1f}%', linewidth=2)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Ajouter les valeurs
for bar, val in zip(bars2, taux):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('analyse_sensibilite.png', dpi=300, bbox_inches='tight')
print("   ✅ Graphique sauvegardé : analyse_sensibilite.png")

# ============================================================================
# 5. SAUVEGARDER LES RÉSULTATS
# ============================================================================
resultats = {
    'impacts': {
        'revision': float(impact_rev * 100),
        'confiance': float(impact_conf * 100),
        'sante': float(impact_mal * 100)
    },
    'classement': [
        {'rang': i+1, 'facteur': f, 'impact': float(imp*100)}
        for i, (f, imp) in enumerate(impacts_sorted)
    ],
    'facteur_principal': facteur_principal,
    'taux_par_condition': {
        'revision_oui': float(rev_oui * 100),
        'revision_non': float(rev_non * 100),
        'confiance_oui': float(conf_oui * 100),
        'confiance_non': float(conf_non * 100),
        'sante_bonne': float(mal_non * 100),
        'sante_malade': float(mal_oui * 100)
    },
    'scenarios_extremes': {
        'meilleur_cas': float(best['Succes'].mean() * 100) if len(best) > 0 else None,
        'pire_cas': float(worst['Succes'].mean() * 100) if len(worst) > 0 else None
    }
}

with open('analyse_sensibilite.json', 'w', encoding='utf-8') as f:
    json.dump(resultats, f, indent=2, ensure_ascii=False)

print("   ✅ Résultats sauvegardés : analyse_sensibilite.json")

# ============================================================================
# CONCLUSION
# ============================================================================
print(f"\n{'='*70}")
print("✅ ANALYSE TERMINÉE")
print("="*70)
print(f"\n📌 CONCLUSION :")
print(f"   Le facteur '{facteur_principal}' est le plus déterminant")
print(f"   avec un impact de {impacts_sorted[0][1]*100:.1f} points sur la réussite.")
print(f"\n💡 RECOMMANDATION :")
print(f"   Prioriser les actions sur : {' > '.join([f[0] for f in impacts_sorted])}")
print("\n" + "="*70)