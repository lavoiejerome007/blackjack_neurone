import os
import shutil
from ultralytics import YOLO

def lancer_entrainement():
    print("🚀 Chargement du modèle de base YOLOv8n...")
    model = YOLO('yolov8n.pt')

    chemin_yaml = 'PokerTable.v11i.yolov8/data.yaml'

    if not os.path.exists(chemin_yaml):
        print(f"❌ Erreur : Le fichier '{chemin_yaml}' est introuvable.")
        return

    print(f"🎯 Fichier de configuration détecté : {chemin_yaml}")
    print("🚀 Lancement de l'entraînement YOLOv8 (cela peut prendre quelques minutes)...")
    
    # Lancement de l'entraînement
    model.train(
        data=chemin_yaml, 
        epochs=25,          
        imgsz=640,          
        batch=4
    )

    print("\n🔍 Recherche du fichier de poids généré...")
    
    # Recherche automatique du fichier best.pt dans le dossier 'runs'
    source_best = None
    for root, dirs, files in os.walk('runs'):
        if 'best.pt' in files:
            source_best = os.path.join(root, 'best.pt')

    if source_best and os.path.exists(source_best):
        destination = 'card_detector.pt'
        shutil.copy(source_best, destination)
        print(f"\n✨ Succès total ! Le modèle a été copié et renommé en : '{destination}'")
        print("📂 Tu le vois maintenant directement dans ton explorateur de fichiers à gauche !")
    else:
        print("❌ Erreur : Impossible de retrouver le fichier best.pt généré.")

if __name__ == '__main__':
    lancer_entrainement()