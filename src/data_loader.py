import os
import cv2
import numpy as np

def load_dataset(base_path: str):
    """
    Carrega imagens de múltiplas pastas (uma por pessoa) e as transforma em vetores.
    Args:
        base_path (str): Caminho da pasta base (training ou testing)

    Returns:
        Tuple[np.ndarray, List[int]]: Matriz de imagens (cada linha é uma imagem flatten) e lista de rótulos
    """
    images = []
    labels = []

    for i, person_folder in enumerate(sorted(os.listdir(base_path))):
        person_path = os.path.join(base_path, person_folder)

        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            images.append(image.flatten())
            labels.append(i)  # o índice da pasta vai ser o rótulo

    return np.array(images), np.array(labels)
