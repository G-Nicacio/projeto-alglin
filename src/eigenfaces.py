import numpy as np

def compute_pca(X: np.ndarray, num_components: int = 50):
    """
    Calcula os autovetores e autovalores para PCA com base nas imagens.
    
    Args:
        X (np.ndarray): Matriz normalizada (amostras x pixels)
        num_components (int): Número de componentes principais (eigenfaces)

    Returns:
        tuple: (media, autovalores, autovetores, projeções)
    """
    # Calcula a média de cada pixel
    mean_face = np.mean(X, axis=0)
    
    # Centraliza os dados
    X_centered = X - mean_face

    # Calcula a matriz de covariância
    covariance_matrix = np.dot(X_centered, X_centered.T)

    # Autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Ordena do maior para o menor
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Seleciona os principais autovetores
    eigenvectors = eigenvectors[:, :num_components]

    # Projeta os autovetores para o espaço original
    eigenfaces = np.dot(X_centered.T, eigenvectors)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)  # normaliza

    # Projeta as imagens no novo espaço
    projections = np.dot(X_centered, eigenfaces)

    return mean_face, eigenfaces, projections
