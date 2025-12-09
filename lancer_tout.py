import subprocess
import sys
import os

print("="*60)
print("🎓 SYSTÈME DE PRÉDICTION - LANCEMENT AUTOMATIQUE")
print("="*60)

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {script_name} terminé avec succès !")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur dans {script_name}")
        print(f"Code erreur: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Fichier {script_name} introuvable !")
        return False

# Étape 1 : Génération des données
success1 = run_script('generate_data.py', 'ÉTAPE 1/2 - Génération des données')

if not success1:
    print("\n⚠️  Abandon : Impossible de générer les données")
    sys.exit(1)

# Étape 2 : Entraînement du modèle
success2 = run_script('train_bayesian_model_simple.py', 'ÉTAPE 2/2 - Entraînement du modèle')

if not success2:
    print("\n⚠️  Abandon : Impossible d'entraîner le modèle")
    sys.exit(1)

# Vérification des fichiers
print(f"\n{'='*60}")
print("📋 VÉRIFICATION DES FICHIERS")
print(f"{'='*60}\n")

required_files = [
    'ma_base_donnees.csv',
    'bayesian_model_simple.joblib',
    'model_probabilities.json',
    'index.html'
]

all_ok = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file} ({size} bytes)")
    else:
        print(f"   ❌ {file} MANQUANT")
        all_ok = False

print(f"\n{'='*60}")
if all_ok:
    print("🎉 SUCCÈS ! Tous les fichiers sont prêts !")
    print("{'='*60}\n")
    print("📌 PROCHAINE ÉTAPE :")
    print("   → Ouvrez 'index.html' dans votre navigateur")
    print("   → Ou utilisez : python -m http.server 8000")
    print("   → Puis allez sur : http://localhost:8000")
else:
    print("⚠️  ATTENTION : Certains fichiers manquent")
    print("{'='*60}")

print("\n" + "="*60)
print("🏁 PROCESSUS TERMINÉ")
print("="*60)