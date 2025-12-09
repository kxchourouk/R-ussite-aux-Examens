import subprocess
import sys
import os

print("="*80)
print(" 🎓 SYSTÈME DE PRÉDICTION RÉUSSITE ÉTUDIANTE - PROJET COMPLET")
print("="*80)
print("\nCe script va :")
print("  1️⃣  Générer les données simulées")
print("  2️⃣  Entraîner le modèle bayésien")
print("  3️⃣  Analyser la sensibilité des facteurs")
print("  4️⃣  Valider les performances du modèle")
print("  5️⃣  Lancer le serveur web")
print("\n" + "="*80)

input("\nAppuyez sur Entrée pour commencer...")

def run_script(script_name, description, obligatoire=True):
    """Exécute un script Python"""
    print(f"\n{'='*80}")
    print(f"▶️  {description}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(script_name):
        if obligatoire:
            print(f"❌ ERREUR : {script_name} introuvable !")
            return False
        else:
            print(f"⚠️  {script_name} introuvable (optionnel, passage à la suite)")
            return True
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {description} terminé avec succès !")
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ Erreur dans {script_name}")
        return False

# ============================================================================
# ÉTAPE 1 : Génération des données
# ============================================================================
if not run_script('generate_data.py', '1️⃣  Génération des données', obligatoire=True):
    print("\n⚠️  Impossible de continuer sans données")
    input("\nAppuyez sur Entrée pour quitter...")
    sys.exit(1)

# ============================================================================
# ÉTAPE 2 : Entraînement du modèle
# ============================================================================
if not run_script('train_bayesian_model_simple.py', '2️⃣  Entraînement du modèle bayésien', obligatoire=True):
    print("\n⚠️  Impossible de continuer sans modèle")
    input("\nAppuyez sur Entrée pour quitter...")
    sys.exit(1)

# ============================================================================
# ÉTAPE 3 : Analyse de sensibilité
# ============================================================================
run_script('analyse_sensibilite.py', '3️⃣  Analyse de sensibilité', obligatoire=False)

# ============================================================================
# ÉTAPE 4 : Validation du modèle
# ============================================================================
run_script('validation_modele.py', '4️⃣  Validation des performances', obligatoire=False)

# ============================================================================
# VÉRIFICATION FINALE
# ============================================================================
print(f"\n{'='*80}")
print("📋 VÉRIFICATION DES FICHIERS GÉNÉRÉS")
print("="*80)

fichiers = {
    'ma_base_donnees.csv': ('Données d\'entraînement', True),
    'bayesian_model_simple.joblib': ('Modèle Python', True),
    'model_probabilities.json': ('Modèle JSON', True),
    'index.html': ('Interface web', True),
    'analyse_sensibilite.png': ('Graphique sensibilité', False),
    'analyse_sensibilite.json': ('Résultats sensibilité', False),
    'validation_modele.png': ('Graphique validation', False),
    'validation_resultats.json': ('Résultats validation', False)
}

tout_ok = True
fichiers_optionnels_manquants = []

for fichier, (description, obligatoire) in fichiers.items():
    if os.path.exists(fichier):
        taille = os.path.getsize(fichier)
        print(f"   ✅ {fichier:<35} ({taille:>8,} bytes) - {description}")
    else:
        if obligatoire:
            print(f"   ❌ {fichier:<35} MANQUANT - {description}")
            tout_ok = False
        else:
            print(f"   ⚠️  {fichier:<35} ABSENT   - {description} (optionnel)")
            fichiers_optionnels_manquants.append(fichier)

# ============================================================================
# RAPPORT FINAL
# ============================================================================
print(f"\n{'='*80}")
print("📊 RAPPORT FINAL")
print("="*80)

if tout_ok:
    print("\n✅ TOUS LES FICHIERS ESSENTIELS SONT PRÉSENTS !")
    
    if fichiers_optionnels_manquants:
        print(f"\n⚠️  {len(fichiers_optionnels_manquants)} fichiers optionnels manquants :")
        for f in fichiers_optionnels_manquants:
            print(f"    • {f}")
        print("\n💡 Assurez-vous que les scripts d'analyse existent et ont bien tourné")
    else:
        print("\n🌟 TOUS LES FICHIERS (y compris optionnels) SONT PRÉSENTS !")
        print("🏆 PROJET 100% COMPLET !")
    
    print(f"\n{'='*80}")
    print("🎯 PROCHAINES ÉTAPES :")
    print("="*80)
    print("\n1. Lancer le serveur web :")
    print("   → python lancer_serveur.py")
    print("   → Ou : python -m http.server 8000")
    print("\n2. Ouvrir dans le navigateur :")
    print("   → http://localhost:8000")
    print("\n3. Pour la présentation, montrer :")
    print("   ✓ Interface web (index.html)")
    print("   ✓ Graphique sensibilité (analyse_sensibilite.png)")
    print("   ✓ Graphique validation (validation_modele.png)")
    print("   ✓ Terminal avec les statistiques")
    
else:
    print("\n❌ ERREUR : Des fichiers essentiels manquent")
    print("\n💡 Solutions :")
    print("   • Vérifiez que tous les scripts sont présents")
    print("   • Relancez ce script")

print("\n" + "="*80)
print("🏁 PROCESSUS TERMINÉ")
print("="*80)

# Demander si l'utilisateur veut lancer le serveur
print("\n" + "="*80)
reponse = input("Voulez-vous lancer le serveur web maintenant ? (o/n) : ").lower()

if reponse in ['o', 'oui', 'y', 'yes']:
    print("\n🚀 Lancement du serveur...")
    try:
        subprocess.run([sys.executable, 'lancer_serveur.py'])
    except FileNotFoundError:
        print("\n⚠️  lancer_serveur.py introuvable")
        print("Lancement manuel :")
        subprocess.run([sys.executable, '-m', 'http.server', '8000'])
else:
    print("\n👍 OK ! Lancez-le manuellement quand vous voulez :")
    print("   python lancer_serveur.py")