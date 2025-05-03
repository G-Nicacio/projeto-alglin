import numpy as np

def preprocess_images(X: np.ndarray) -> np.ndarray:
    """
    Normaliza as imagens: média 0, desvio padrão 1.
    
    Args:
        X (np.ndarray): Matriz de imagens (amostras x pixels)
    
    Returns:
        np.ndarray: Matriz normalizada
    """
    X = X.astype('float32')
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Evita divisão por zero
    std[std == 0] = 1

    return (X - mean) / std
