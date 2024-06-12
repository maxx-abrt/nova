import streamlit as st
import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json

# Charger le modèle pré-entraîné DenseNet121
model = xrv.models.DenseNet(weights="densenet121-res224-all")

# Dictionnaire de traduction des affections en français
translations = {
    "Atelectasis": "Atélectasie",
    "Cardiomegaly": "Cardiomégalie",
    "Consolidation": "Consolidation",
    "Edema": "Œdème",
    "Effusion": "Épanchement",
    "Emphysema": "Emphysème",
    "Fibrosis": "Fibrose",
    "Hernia": "Hernie",
    "Infiltration": "Infiltration",
    "Mass": "Masse",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Épaississement pleural",
    "Pneumonia": "Pneumonie",
    "Pneumothorax": "Pneumothorax",
    "No Finding": "Aucune affection",
    "Lung Lesion": "Lésion pulmonaire",
    "Fracture": "Fracture",
    "Lung Opacity": "Opacité pulmonaire",
    "Enlarged Cardiomediastinum": "Cardiomédiastin élargi"
}

# Fonction pour prédire l'affection à partir d'une image
def predict(image_path):
    img = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    img = img.resize((224, 224))  # Redimensionner l'image
    img = np.array(img) / 255.0  # Normaliser l'image dans la plage [0, 1]
    img = img * 2048 - 1024  # Normaliser l'image dans la plage [-1024, 1024]
    img = img[None, None, :, :]  # Ajouter les dimensions de batch et de canal
    img = torch.from_numpy(img).float()
    with torch.no_grad():
        outputs = model(img)
    return dict(zip(model.pathologies, outputs[0].numpy())), img

# Fonction pour générer une carte de chaleur Grad-CAM
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    img_tensor.requires_grad = True
    outputs = model(img_tensor)
    score = outputs[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    gradients = model.features.denseblock4.denselayer16.conv2.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.features.denseblock4.denselayer16.conv2.weight.detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)
    return heatmap.numpy()

# Fonction pour superposer une grille sur une image et afficher les numéros des grilles
def overlay_grid(image, grid_size=(10, 10)):
    h, w = image.shape[:2]
    step_x, step_y = w // grid_size[0], h // grid_size[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = i * step_x
            y = j * step_y
            cv2.rectangle(image, (x, y), (x + step_x, y + step_y), color, thickness)
            cv2.putText(image, f"{chr(65 + i)}{j + 1}", (x + 5, y + 15), font, font_scale, color, thickness)
    return image

# Interface utilisateur avec Streamlit
st.set_page_config(page_title="Classification de Radiographies Thoraciques - Nova - Modèle 1 V.1 - Max Aubert.", layout="wide")
st.title("Classification de Radiographies Thoraciques - Nova - Max Aubert")

uploaded_file = st.file_uploader("Choisissez une image...", type="png")
if uploaded_file is not None:
    patient_name = st.text_input("Nom du patient")
    file_name = st.text_input("Nom du fichier")
    if st.button("Analyser"):
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée.', use_column_width=True)
        st.write("")
        st.write("Classification en cours...")
        image_path = f"{file_name}.png"
        image.save(image_path)
        start_time = time.time()
        predictions, img_tensor = predict(image_path)
        execution_time = time.time() - start_time
        st.subheader("Résultats des Prédictions")
        st.write("Les probabilités des affections détectées sont les suivantes :")
        for condition, probability in predictions.items():
            condition_fr = translations.get(condition, condition)
            st.write(f"{condition_fr} : {probability * 100:.2f}%")
            st.progress(float(probability))
        most_probable_condition = max(predictions, key=predictions.get)
        most_probable_probability = predictions[most_probable_condition]
        if all(prob < 0.1 for prob in predictions.values()):
            st.write("La radiographie semble indiquer une situation saine, sans affection détectée.")
        else:
            most_probable_condition_fr = translations.get(most_probable_condition, most_probable_condition)
            st.write(f"La condition la plus probable est : {most_probable_condition_fr} avec une probabilité de {most_probable_probability * 100:.2f}%.")
            st.write(f"Cette prédiction est la plus probable car le modèle a détecté des caractéristiques dans l'image qui correspondent à {most_probable_condition_fr}.")
            target_class = model.pathologies.index(most_probable_condition)
            heatmap = generate_gradcam(model, img_tensor, target_class)
            img = cv2.imread(image_path)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            superimposed_img = cv2.addWeighted(img_gray, 0.6, heatmap, 0.4, 0)
            superimposed_img_with_grid = overlay_grid(superimposed_img)
            st.image(superimposed_img_with_grid, caption='Carte de chaleur Grad-CAM avec grille', use_column_width=True, clamp=True)
            st.write("### Explication de la Prédiction")
            st.write(f"Le modèle a identifié des caractéristiques spécifiques dans l'image qui sont typiques de {most_probable_condition_fr}.")
            st.write(f"Par exemple, pour la {most_probable_condition_fr}, le modèle peut avoir détecté des opacités,...")
            st.write("### Marge d'erreur")
            error_margin = 1 - most_probable_probability
            st.write(f"Le modèle estime une marge d'erreur de {error_margin * 100:.2f}%.")
            st.progress(error_margin)
            st.write("### Visualisation des résultats")
            st.bar_chart([prob * 100 for prob in predictions.values()], height=300)
            st.write("### Détails des affections")
            for condition, probability in predictions.items():
                condition_fr = translations.get(condition, condition)
                st.write(f"**{condition_fr}** : {probability * 100:.2f}%")
            st.write("### Temps d'exécution")
            st.write(f"Le temps d'exécution pour la classification est de {execution_time:.2f} secondes.")
