import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    # Caminho do CSV e da pasta de saída
    csv_path = 'results/recognition_results.csv'
    plots_path = 'results/plots'
    os.makedirs(plots_path, exist_ok=True)

    # Lê os resultados
    df = pd.read_csv(csv_path)
    y_true = df['True Label']
    y_pred = df['Predicted Label']
    distances = df['Distance']

    # ======= 1. Matriz de Confusão =======
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(f'{plots_path}/confusion_matrix.png')
    plt.close()

    # ======= 2. Acurácia por Classe =======
    df['Correct'] = y_true == y_pred
    acc_by_class = df.groupby('True Label')['Correct'].mean() * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x=acc_by_class.index, y=acc_by_class.values, palette='viridis', legend=False)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    plt.title('Accuracy by Class')
    plt.ylim(0, 100)
    plt.savefig(f'{plots_path}/accuracy_per_class.png')
    plt.close()

    # ======= 3. Histograma de distâncias =======
    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=20, kde=True, color='dodgerblue')
    plt.xlabel('Distance to Closest Match')
    plt.ylabel('Frequency')
    plt.title('Distribution of Recognition Distances')
    plt.savefig(f'{plots_path}/distance_distribution.png')
    plt.close()

    # ======= 4. Boxplot de distância por classe real =======
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='True Label', y='Distance', data=df, palette='coolwarm')
    plt.title('Distance Distribution per Class')
    plt.savefig(f'{plots_path}/distance_boxplot.png')
    plt.close()

    print("Todos os gráficos foram salvos em:", plots_path)

if __name__ == "__main__":
    main()
