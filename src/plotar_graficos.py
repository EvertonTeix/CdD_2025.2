import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plotar_matriz_correlacao(dataframe,figsize=(15, 15),annot=True,cmap="viridis",titulo="Matriz de Correlação"):
    corr = dataframe.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap)
    plt.title(titulo)
    plt.show()

def gerar_e_salvar_shap_bar_duas_classes(X, y):
    # Treino/teste (mesma lógica do seu)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    base_path = Path("../../../reports/figures/SHAP_diabetes")
    base_path.mkdir(parents=True, exist_ok=True)

    # Explainer
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        # Se for lista, pegamos os elementos direto
        sv_class0 = shap_values[0]
        sv_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Se for 3D, a última dimensão são as classes
        sv_class0 = shap_values[:, :, 0]
        sv_class1 = shap_values[:, :, 1]
    else:
        # Caso seja binário simplificado (apenas um array 2D)
        sv_class0 = shap_values
        sv_class1 = shap_values

    # Função auxiliar para plotar sem erro de shape
    def salvar_plot(sv, nome_arquivo):
        plt.figure(figsize=(10, 6))
        # Garantimos que passamos o sv e o X_test.values para não ter conflito de índice/coluna
        shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        caminho = base_path / nome_arquivo
        plt.savefig(caminho, bbox_inches='tight')
        plt.close()
        print(f"Salvo: {caminho}")

    salvar_plot(sv_class0, "SHAP_bar.png")


def plot_matriz_confusao(y_true, y_pred):
    classes = [0, 1]
    cm = confusion_matrix(y_true, y_pred)  # números absolutos
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusão - Modelo Final")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
    
def plot_matriz_por_tecnica_modelo(resultados, tecnica, modelo):
    y_true_all = []
    y_pred_all = []

    folds = resultados[tecnica][modelo]

    for f in folds:
        y_true_all.extend(f['y_true'])
        y_pred_all.extend(f['y_pred'])

    plot_matriz_confusao(y_true_all, y_pred_all)
