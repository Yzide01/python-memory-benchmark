import subprocess
import time

# Liste des scripts de plotting que tu as créés
plots = [
    "plot-grpby-pc.py",
    "plot-join-pc.py"
]

def run_all():
    print("--- Démarrage de la suite complète de benchmarks ---")
    
    for plot_script in plots:
        print(f"\n[ORCHESTRATEUR] Lancement de : {plot_script}")
        try:
            # On appelle le script de plot qui s'occupe de tout (collecte + graph)
            subprocess.run(["python3", plot_script], check=True)
            print(f"[OK] {plot_script} terminé.")
        except subprocess.CalledProcessError:
            print(f"[ERREUR] Le script {plot_script} a échoué.")
            break
        
        # Petite pause pour laisser le CPU refroidir un peu entre les runs (évite le thermal throttling)
        time.sleep(2)

    print("\n--- Tous les graphiques ont été générés avec succès ! ---")

if __name__ == "__main__":
    run_all()
