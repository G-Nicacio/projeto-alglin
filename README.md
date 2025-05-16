# Reconhecimento Facial com Eigenfaces (Projeto de Álgebra Linear)

Este repositório contém um projeto completo de reconhecimento facial baseado na técnica de **Eigenfaces**, implementado do zero como trabalho prático para a disciplina de Álgebra Linear. O projeto utiliza **PCA (Análise de Componentes Principais)** para extrair características relevantes de rostos humanos e comparar novas imagens com uma base de dados de referência.

---

## 📁 Estrutura do Repositório

```bash
.
├── main.py                 # Executa o pipeline completo de treino, teste e exportação de CSV
├── plot_results.py         # Gera gráficos de análise (acurácia, distância por classe, etc.)
├── requirements.txt        # Dependências do projeto
├── src/                    # Lógica central do sistema
│   ├── data_loader.py      # Funções para carregar datasets
│   ├── preprocessing.py    # Pré-processamento (normalização e reshape)
│   ├── eigenfaces.py       # Cálculo de média, PCA, projeções
│   └── recognizer.py       # Classe que faz o reconhecimento propriamente dito
├── data/                   # Dataset dividido entre treino e teste
│   ├── training/
│   └── testing/
├── results/                # Resultados gerados (CSV + gráficos)
│   ├── recognition_results.csv
│   └── plots/

```

---

## 🧠 O que é Eigenfaces?

A técnica de **Eigenfaces** representa rostos humanos como combinações lineares de "faces-base", geradas a partir de autovetores da matriz de covariância das imagens de treino. Isso permite reduzir drasticamente a dimensionalidade do problema sem perder as características mais relevantes.

---

## 📌 Como Executar

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Execute o reconhecimento facial completo:
```bash
python main.py
```

Isso vai:
- Carregar as imagens de treino e teste
- Aplicar PCA
- Reconhecer cada imagem de teste
- Gerar o CSV `results/recognition_results.csv`
- Imprimir a acurácia e distâncias

### 3. Gerar gráficos:
```bash
python plot_results.py
```

## 📊 Resultados Obtidos

- Acurácia: **96,67%**
- Média de distância (match correto): **42.3**
- Média de distância (match incorreto): **85.7**

Esses valores indicam que o sistema é eficaz em distinguir corretamente entre rostos diferentes e semelhantes.

---

## 📚 Referências

- Turk, M., & Pentland, A. (1991). *Eigenfaces for recognition*. Journal of Cognitive Neuroscience.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
- Jolliffe, I. T. (2002). *Principal Component Analysis*.

---

## 🧠 Créditos

As imagens de rosto utilizadas para treinamento e teste foram obtidas de bases públicas para fins educacionais, como a [Yale Face Database](http://vision.ucsd.edu/content/yale-face-database) e a [ORL AT&T Face Database](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

Este projeto foi desenvolvido por **Gustavo Nicacio** como trabalho da disciplina de Álgebra Linear. A estética do vídeo e a narrativa se inspiram no estilo dos vídeos do **Vsauce**, misturando ciência e humor.