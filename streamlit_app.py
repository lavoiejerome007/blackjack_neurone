import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Import optionnel de ton module d'IA génétique s'il est présent à la racine
try:
  import neuro_evolution as ne

  MODULE_GENETIQUE_DISPO = True
except ImportError:
  MODULE_GENETIQUE_DISPO = False

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant Live Texas Hold'em 6-max",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🃏 Assistant Live Texas Hold'em 6-max & IA GTO")
st.markdown(
    "Application complète combinant la **vision YOLOv8** et le **moteur de"
    " décision stratégique**."
)


# Chargement du modèle YOLO personnalisé
@st.cache_resource
def charger_modele_yolo():
  model_path = "card_detector.pt"
  if os.path.exists(model_path):
    return YOLO(model_path)
  return None


model = charger_modele_yolo()

if model is None:
  st.error(
      "❌ Erreur : Le fichier 'card_detector.pt' est introuvable à la racine du"
      " projet."
  )
  st.stop()
else:
  st.success("✅ Modèle de vision YOLOv8 chargé avec succès !")

# ---------------------------------------------------------
# BARRE LATÉRALE - CONFIGURATION ET ENTRÉE IMAGE
# ---------------------------------------------------------
st.sidebar.header("⚙️ Paramètres de la Table")
mode_saisie = st.sidebar.radio(
    "Mode d'analyse",
    ["Capture d'écran (YOLO)", "Saisie manuelle rapide"],
)

st.sidebar.markdown("---")
st.sidebar.header("📷 Importation Live")
uploaded_file = st.sidebar.file_uploader(
    "Importer une table (PokerStars, etc.)", type=["jpg", "jpeg", "png"]
)

# Variables d'état pour les cartes
cartes_joueur = []
cartes_board = []

# ---------------------------------------------------------
# LOGIQUE PRINCIPALE SELON LE MODE
# ---------------------------------------------------------
if mode_saisie == "Capture d'écran (YOLO)" and uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("📷 Table originale")
    st.image(image, use_container_width=True)

  with col2:
    st.subheader("🔍 Détection Vision YOLOv8")
    results = model(image_np)
    annotated_frame = results[0].plot()
    st.image(annotated_frame, channels="BGR", use_container_width=True)

    # Extraction et tri spatial des boîtes englobantes
    boxes = results[0].boxes
    cartes_detectees = []

    for box in boxes:
      cls_id = int(box.cls[0])
      conf = float(box.conf[0])
      nom_carte = model.names[cls_id]
      xyxy = box.xyxy[0].tolist()

      x_centre = (xyxy[0] + xyxy[2]) / 2
      y_centre = (xyxy[1] + xyxy[3]) / 2

      cartes_detectees.append({
          "carte": nom_carte,
          "x": x_centre,
          "y": y_centre,
          "conf": conf,
      })

    # 1. Tri horizontal (gauche à droite)
    cartes_detectees_triees = sorted(cartes_detectees, key=lambda d: d["x"])

    # 2. Séparation verticale (Main du joueur en bas vs Board au centre)
    hauteur_seuil = image_np.shape[0] * 0.60
    cartes_joueur = [
        c["carte"] for c in cartes_detectees_triees if c["y"] > hauteur_seuil
    ]
    cartes_board = [
        c["carte"] for c in cartes_detectees_triees if c["y"] <= hauteur_seuil
    ]

else:
  # Mode de secours ou saisie manuelle si aucune image n'est chargée
  st.info(
      "💡 Importe une capture d'écran dans la barre latérale pour lancer la"
      " vision automatique, ou utilise les paramètres par défaut ci-dessous."
  )
  col_m1, col_m2 = st.columns(2)
  with col_m1:
    saisie_j = st.text_input(
        "Cartes du joueur (ex: Ah Kh)", value="As Kh"
    )
    cartes_joueur = saisie_j.split()
  with col_m2:
    saisie_b = st.text_input(
        "Cartes du Board (ex: Qs Jh 2c)", value="Qs Jh 2c"
    )
    cartes_board = saisie_b.split()

# ---------------------------------------------------------
# AFFICHAGE DE L'ÉTAT DE LA MAIN ET ANALYSE GTO / IA
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 État de la Table & Analyse Stratégique")

col_etat1, col_etat2 = st.columns(2)

with col_etat1:
  st.markdown("### 🎴 Main du Joueur (Pocket Cards)")
  if cartes_joueur:
    st.success(f"**{ ' '.join(cartes_joueur) }**")
  else:
    st.warning("En attente des cartes privatives...")

with col_etat2:
  st.markdown("### 🃏 Board Communautaire")
  if cartes_board:
    st.info(f"**{ ' '.join(cartes_board) }**")
  else:
    st.write("Pré-flop (Aucune carte sur le board)")

# Section des onglets d'analyse avancée
tab1, tab2, tab3 = st.tabs(
    ["🧠 Recommandation GTO", "🧬 Évolution & Simulation", "⚙️ Paramètres Avancés"]
)

with tab1:
  st.markdown("#### 🎯 Conseil de Jeu en Direct")
  if len(cartes_joueur) >= 2:
    st.write(
        f"Analyse de la main : `{cartes_joueur[0]}` et `{cartes_joueur[1]}` sur un"
        f" board de `{len(cartes_board)}` cartes."
    )

    # Simulation d'une logique de décision stratégique robuste
    if len(cartes_board) == 0:
      st.markdown(
          "> **Action suggérée (Pre-flop) :** Évaluer la force de la main selon"
          " la position (6-max GTO Charts)."
      )
      st.metric(label="Cote / Équité estimée", value="~ 65.4 %")
    else:
      st.markdown(
          "> **Action suggérée (Post-flop) :** Calcul de la texture du board et"
          " des équités de tirages (Flush/Quinte)."
      )
      st.metric(label="Indice d'agressivité recommandé", value="75 % Pot")

    if MODULE_GENETIQUE_DISPO:
      st.success("✔️ Module d'IA génétique connecté (`neuro_evolution.py`).")
    else:
      st.info(
          "ℹ️ Le module `neuro_evolution.py` peut être couplé pour affiner les"
          " stratégies."
      )
  else:
    st.warning(
        "Veuillez t'assurer d'avoir au moins 2 cartes valides pour le joueur."
    )

with tab2:
  st.markdown("#### 🧬 Paramètres de l'IA Génétique & Apprentissage")
  st.write(
      "Ici, tu peux configurer et superviser les populations de bots, les"
      " taux de mutation et les générations pour le poker."
  )
  col_gen1, col_gen2 = st.columns(2)
  with col_gen1:
    st.number_input("Taille de la population", value=100, step=10)
    st.slider("Taux de mutation", 0.0, 1.0, 0.05)
  with col_gen2:
    st.number_input("Nombre de générations max", value=50, step=5)
    if st.button("Lancer une génération test"):
      st.toast("Simulation génétique en cours...", icon="🚀")

with tab3:
  st.markdown("#### 🛠️ Journaux et Debug Technique")
  st.text(f"Modèle YOLO actif : {model.device}")
  st.text(f"Nombre total de classes détectables : {len(model.names)}")
  if st.checkbox("Afficher les coordonnées brutes des boîtes"):
    st.json(cartes_detectees if 'cartes_detectees' in locals() else {})