import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_images
from eigenfaces import compute_pca
from recognizer import FaceRecognizer

def main():
    training_path = 'data/training'
    testing_path = 'data/testing'

    print("Carregando imagens de treino...")
    train_images, train_labels = load_dataset(training_path)

    print("Pré-processando imagens de treino...")
    train_images = preprocess_images(train_images)

    print("Treinando modelo de Eigenfaces...")
    mean_face, eigenfaces, train_projections = compute_pca(train_images)

    print("Carregando imagens de teste...")
    test_images, test_labels = load_dataset(testing_path)
    test_images = preprocess_images(test_images)

    print("Iniciando reconhecimento...")
    recognizer = FaceRecognizer(mean_face, eigenfaces, train_projections, train_labels)

    csv_path = os.path.join("results", "recognition_results.csv")
    correct_distances = []
    wrong_distances = []
    acertos = 0

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label", "Distance"])

        for img, true_label in zip(test_images, test_labels):
            predicted_label, distance = recognizer.recognize(img)
            writer.writerow([true_label, predicted_label, f"{distance:.4f}"])
            print(f"Verdadeiro: {true_label} | Predito: ({predicted_label}, {distance:.2f})")

            if predicted_label == true_label:
                acertos += 1
                correct_distances.append(distance)
            else:
                wrong_distances.append(distance)

    total = len(test_labels)
    acc = acertos / total * 100
    mean_correct = sum(correct_distances) / len(correct_distances) if correct_distances else 0
    mean_wrong = sum(wrong_distances) / len(wrong_distances) if wrong_distances else 0

    print(f"\n Acurácia: {acertos}/{total} ({acc:.2f}%)")
    print(f"Média de distância (acertos): {mean_correct:.2f}")
    print(f"Média de distância (erros): {mean_wrong:.2f}")
    print(f"Resultados salvos em: {csv_path}")

if __name__ == "__main__":
    main()
