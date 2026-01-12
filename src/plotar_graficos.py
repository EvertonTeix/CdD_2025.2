import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

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
    
def plot_matriz_por_tecnica_modelo(resultados, tecnica, configuracao, modelo):
    y_true_all = []
    y_pred_all = []
    try:
        folds = resultados[tecnica][configuracao][modelo]
    except KeyError:
        print(f"Erro: Não encontrei a combinação [{tecnica}] -> [{configuracao}] -> [{modelo}]")
        return

    for f in folds:
        y_true_all.extend(f['y_true'])
        y_pred_all.extend(f['y_pred'])

    plot_matriz_confusao(y_true_all, y_pred_all)

def plot_pizza_sentimentos(df, coluna, titulo="Distribuição de Sentimentos"):

    counts = df[coluna].value_counts()
    labels = counts.index
    sizes = counts.values

    plt.figure(figsize=(6,6))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=sns.color_palette("pastel")
    )
    plt.title(titulo)
    plt.show()
    
    
def plot_histograma_comprimento_texto(df, coluna_texto, titulo="Distribuição do número de palavras", bins=30, kde=True):
    comprimento = df[coluna_texto].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(8, 5))
    sns.histplot(comprimento, bins=bins, kde=kde)
    plt.xlabel("Número de palavras")
    plt.ylabel("Frequência")
    plt.title(titulo)
    plt.show()
    
def plot_wordcloud_por_categoria(df, coluna_texto, coluna_categoria, largura=800, altura=400):
    categorias = df[coluna_categoria].unique()
    
    for cat in categorias:
        text = " ".join(df[df[coluna_categoria] == cat][coluna_texto].astype(str))
        
        wordcloud = WordCloud(width=largura, height=altura, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuvem de Palavras - {cat}")
        plt.show()
        

def plot_top_tfidf(X, feature_names, top_n=20, titulo="Top palavras com maior TF-IDF médio", cor='skyblue'):
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    
    sorted_idx = np.argsort(mean_tfidf)[-top_n:]
    
    plt.figure(figsize=(10, 5))
    plt.barh([feature_names[i] for i in sorted_idx], mean_tfidf[sorted_idx], color=cor)
    plt.xlabel("TF-IDF médio")
    plt.ylabel("Palavras")
    plt.title(titulo)
    plt.show()
    
def plot_ranking_experimentos(df_resultados):
    ranking = df_resultados.groupby(['tecnica', 'configuracao', 'modelo'])['f1'].agg(['mean', 'std']).reset_index()
    ranking.columns = ['Técnica', 'Configuração', 'Modelo', 'F1-Médio', 'Desvio-Padrão']
    ranking = ranking.sort_values(by='F1-Médio', ascending=False)

    def simplificar_config(cfg):
        cfg_str = str(cfg)
        ngram = "Unigrama" if "(1, 1)" in cfg_str else "Bigrama"
        features = "10k feat" if "10000" in cfg_str else ""
        return f"{ngram} {features}".strip()

    ranking['Config_Curta'] = ranking['Configuração'].apply(simplificar_config)
    ranking['Label'] = ranking['Modelo'] + " (" + ranking['Config_Curta'] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    tecnicas = ranking['Técnica'].unique()

    for i, tecnica in enumerate(tecnicas):
            dados_plot = ranking[ranking['Técnica'] == tecnica]
            sns.barplot(
                data=dados_plot, 
                x='F1-Médio', 
                y='Label', 
                hue='Label',
                ax=axes[i], 
                palette="viridis" if tecnica == "TF-IDF" else "magma",
                legend=False 
            )
            axes[i].set_title(f'Performance: {tecnica}')
            axes[i].set_xlabel('F1-Score Médio')
            axes[i].set_ylabel('')
            
            axes[i].set_xlim(ranking['F1-Médio'].min() - 0.05, ranking['F1-Médio'].max() + 0.02)
            axes[i].grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return ranking

def plot_boxplot_comprimento_por_categoria(df, coluna_texto, coluna_categoria, titulo="Distribuição do Número de Palavras por Categoria"):
    df_temp = df.copy()
    df_temp["word_count"] = df_temp[coluna_texto].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 5))
    
    sns.boxplot(
        x=coluna_categoria, 
        y="word_count", 
        data=df_temp, 
        hue=coluna_categoria, 
        palette="Set2", 
        legend=False
    )
    
    plt.xlabel(coluna_categoria.capitalize())
    plt.ylabel("Número de palavras")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()