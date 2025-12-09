import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

# Vérifier que index.html existe
if not os.path.exists('index.html'):
    print("❌ ERREUR : index.html introuvable !")
    print("   Assurez-vous d'être dans le bon répertoire.")
    sys.exit(1)

print("="*60)
print("🌐 SERVEUR WEB - Interface de Prédiction")
print("="*60)

Handler = http.server.SimpleHTTPRequestHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n✅ Serveur démarré avec succès !")
        print(f"📍 URL : http://localhost:{PORT}")
        print(f"\n📌 Instructions :")
        print(f"   1. Ouvrez votre navigateur")
        print(f"   2. Allez sur : http://localhost:{PORT}")
        print(f"   3. Utilisez l'interface de prédiction")
        print(f"\n⚠️  Pour arrêter le serveur : Ctrl+C")
        print("="*60)
        
        # Ouvrir automatiquement le navigateur
        try:
            webbrowser.open(f'http://localhost:{PORT}')
            print("\n🌐 Ouverture du navigateur...")
        except:
            print("\n⚠️  Impossible d'ouvrir le navigateur automatiquement")
            print(f"   Ouvrez manuellement : http://localhost:{PORT}")
        
        print("\n🔄 Serveur en cours d'exécution...\n")
        httpd.serve_forever()
        
except KeyboardInterrupt:
    print("\n\n⏹️  Serveur arrêté par l'utilisateur")
    print("="*60)
    sys.exit(0)
except OSError as e:
    if "Address already in use" in str(e):
        print(f"\n❌ ERREUR : Le port {PORT} est déjà utilisé !")
        print(f"\n💡 Solutions :")
        print(f"   1. Fermez l'autre serveur sur le port {PORT}")
        print(f"   2. Ou modifiez PORT dans ce script")
    else:
        print(f"\n❌ ERREUR : {e}")
    sys.exit(1)
