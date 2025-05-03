import numpy as np

class FaceRecognizer:
    def __init__(self, mean_face, eigenfaces, projections, labels):
        """
        Inicializa o reconhecedor facial com base no treinamento.

        Args:
            mean_face (np.ndarray): Média das faces.
            eigenfaces (np.ndarray): Eigenfaces calculadas.
            projections (np.ndarray): Projeções das faces de treino.
            labels (list): Labels correspondentes a cada face de treino.
        """
        self.mean_face = mean_face
        self.eigenfaces = eigenfaces
        self.projections = projections
        self.labels = labels

    def recognize(self, test_image: np.ndarray):
        """
        Reconhece uma imagem projetando-a no espaço das Eigenfaces.

        Args:
            test_image (np.ndarray): Vetor da imagem (flattened).

        Returns:
            tuple: (label mais próximo, distância)
        """
        centered = test_image - self.mean_face
        projection = np.dot(centered, self.eigenfaces)

        distances = np.linalg.norm(self.projections - projection, axis=1)
        min_index = np.argmin(distances)
        return self.labels[min_index], distances[min_index]
