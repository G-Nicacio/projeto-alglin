import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from src.preprocessing import preprocess_images
from src.eigenfaces import compute_pca
from src.recognizer import FaceRecognizer

app = Flask(__name__)

REFERENCE_FOLDER = 'app/faces'
THRESHOLD = 75.0  # limite de distância

# === Carregamento das imagens de referência ===
print("Carregando referências...")
ref_images = []
ref_labels = []
for i, filename in enumerate(os.listdir(REFERENCE_FOLDER)):
    path = os.path.join(REFERENCE_FOLDER, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        ref_images.append(img.flatten())
        ref_labels.append("VOCÊ")

ref_images = np.array(ref_images)
ref_images = preprocess_images(ref_images)
mean_face, eigenfaces, projections = compute_pca(ref_images)
recognizer = FaceRecognizer(mean_face, eigenfaces, projections, ref_labels)

# === Rotas ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    data_url = request.form.get('image_data')
    if not data_url:
        return render_template('index.html', error="Nenhuma imagem foi enviada.")

    # Decodifica imagem base64
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes)).convert('L')

    # Obtém as dimensões reais da imagem de referência
    ref_h_w = ref_images.shape[1]  # total de pixels
    ref_example = ref_images[0].reshape(-1)

    possible_heights = [240, 360, 480, 600]
    for h in possible_heights:
        if ref_h_w % h == 0:
            w = ref_h_w // h
            break
    else:
        raise ValueError("Não foi possível inferir shape da imagem de referência.")

    image = image.resize((w, h))

    image_array = np.array(image).flatten().astype('float32')
    print(f"Imagem: média = {np.mean(image_array):.2f}, desvio = {np.std(image_array):.2f}, shape = {image_array.shape}")
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    # --- NOVO BLOCO: imprime distâncias individuais
    print("\nComparando com todas as referências:")
    for idx, (ref, label) in enumerate(zip(projections, ref_labels)):
        dist = np.linalg.norm(ref - np.dot((image_array - mean_face), eigenfaces))
        print(f"📷 Referência {idx + 1}: {label} | Distância: {dist:.2f}")

    # --- Reconhecimento final
    label, dist = recognizer.recognize(image_array)
    print(f"\nMelhor correspondência: {label} | Distância: {dist:.2f}")

    if label == "VOCÊ" and dist < THRESHOLD:
        return render_template('sala.html')
    else:
        return render_template('index.html', error="Rosto não reconhecido")


@app.route('/sala')
def sala():
    return render_template('sala.html')

if __name__ == '__main__':
    app.run(debug=True)
