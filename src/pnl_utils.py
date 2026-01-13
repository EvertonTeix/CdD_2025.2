
import re
import unicodedata
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.sparse import issparse

stop_words = STOP_WORDS.copy()

palavras_para_remover = {
    "pra", "queria", "oq", "q", "vc", "p", "d", "pq", "dia", "n", "vou", 
    "ta", "tá", "to", "tô", "eh", "oh", "né", "tbm", "tb", "hj", "vcs", "ai", "aí",
    "gente", "pessoa", "pessoas", "coisa", "coisas", "alguém", "ninguém", "mundo",
    "hoje", "agora", "agr", "ano", "semana", "noite", "hora", "lugar",
    "acho", "acha", "vai", "vão", "fazer", "fazendo", "ficar", "falar", "disse", 
    "ir", "ver", "viu", "quer", "ser", "era", "tinha", "tenho", "deu", "bb", "pro", "amo"
}


stop_words.update(palavras_para_remover)

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE) 
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)

    text = ''.join([char for char in text if not unicodedata.category(char).startswith('So')])

    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text.strip()

def salvar_resultados_csv(resultados_gerais, caminho_csv="resultados_modelos.csv"):
    linhas = []

    for tecnica, modelos in resultados_gerais.items():
        for nome_modelo, folds in modelos.items():

            linhas.append({
                "Vetorizador": tecnica,
                "Modelo": nome_modelo,
                "Accuracy": np.mean([f["metricas"]["accuracy"] for f in folds]),
                "Precision": np.mean([f["metricas"]["precision"] for f in folds]),
                "Recall": np.mean([f["metricas"]["recall"] for f in folds]),
                "F1": np.mean([f["metricas"]["f1"] for f in folds]),
                "Melhores_Params": str(folds[0]["best_params"])
            })

    df = (
        pd.DataFrame(linhas)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    df.to_csv(caminho_csv, index=False)
    return df

def resultados_para_dataframe(resultados):
    linhas = []
    for tecnica, configs in resultados.items():
        for nome_config, modelos in configs.items():
            for nome_modelo, folds in modelos.items():
                for info in folds:
                    linhas.append({
                        "tecnica": tecnica,
                        "model": nome_modelo,
                        "fold": info["fold"],
                        "accuracy": info["metricas"]["accuracy"],
                        "precision": info["metricas"]["precision"],
                        "recall": info["metricas"]["recall"],
                        "f1": info["metricas"]["f1"],
                        "best_params": str(info["best_params"]), 
                        "configuracao": nome_config, 
                    })
                    
    return pd.DataFrame(linhas)

def analise_metricas(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary')
    }

def kfold_gridsearch_classificacao(X, y, features_names):

    if isinstance(y, np.ndarray):
        y = pd.Series(y).reset_index(drop=True)
    elif isinstance(y, pd.Series):
        y = y.reset_index(drop=True)

    modelos = {
        "DT": {
            "model": DecisionTreeClassifier,
            "params": {
                "max_depth": range(2, 11),
                "class_weight": ["balanced"]
            }
        },
        "LR": {
            "model": LogisticRegression,
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs"],
                "class_weight": ["balanced"],
                "max_iter": [1000]
            }
        },
        "RF": {
            "model": RandomForestClassifier,
            "params": {
                "n_estimators": [100],
                "max_depth": [None, 10],
                "class_weight": ["balanced"]
            }
        },
        "SVM": {
            "model": SVC,
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear"],
                "class_weight": ["balanced"]
            }
        }
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {nome: [] for nome in modelos}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        print(f"\n===== Fold {fold} =====")
        
        if issparse(X):
            X_train = X[train_idx]
            X_test = X[test_idx]
        else:
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_tr, X_val = X_train, X_test
        y_tr, y_val = y_train, y_test

        scaler_gs = MaxAbsScaler()
        X_tr_sc = scaler_gs.fit_transform(X_tr)
        X_val_sc = scaler_gs.transform(X_val)

        for nome_modelo, info in modelos.items():
            grid = list(ParameterGrid(info["params"]))
            f1_scores = []

            for params in grid:
                model = info["model"](**params)
                model.fit(X_tr_sc, y_tr)
                y_val_pred = model.predict(X_val_sc)

                f1 = f1_score(
                    y_val,
                    y_val_pred,
                    average="binary",
                    zero_division=0
                )
                f1_scores.append(f1)

            best_idx = np.argmax(f1_scores)
            best_params = grid[best_idx]
            best_f1 = f1_scores[best_idx]

            print(
                f"{nome_modelo} | Melhor F1 (val): {best_f1:.4f} | Params: {best_params}"
            )

            scaler_final = MaxAbsScaler()
            X_train_sc = scaler_final.fit_transform(X_train)
            X_test_sc = scaler_final.transform(X_test)

            final_model = info["model"](**best_params)
            final_model.fit(X_train_sc, y_train)

            y_test_pred = final_model.predict(X_test_sc)

            resultados[nome_modelo].append({
                "fold": fold,
                "best_params": best_params,
                "best_f1_val": best_f1,
                "metricas": analise_metricas(y_test, y_test_pred),
                "modelo": final_model,
                "scaler": scaler_final,
                "features_names": features_names,
                "y_true": y_test.tolist(),
                "y_pred": y_test_pred.tolist(),
            })

    return resultados


def tabela_metricas_medias(resultados):
    resumo_list = []

    for tecnica, configs in resultados.items():
        for nome_config, modelos in configs.items():
            for nome_modelo, folds in modelos.items():

                accs, precisions, recalls, f1s = [], [], [], []

                for fold_info in folds:
                    accs.append(fold_info["metricas"]["accuracy"])
                    precisions.append(fold_info["metricas"]["precision"])
                    recalls.append(fold_info["metricas"]["recall"])
                    f1s.append(fold_info["metricas"]["f1"])

                resumo_list.append({
                    "técnica": tecnica,
                    "modelo": nome_modelo,
                    "accuracy_media": np.mean(accs),
                    "precision_media": np.mean(precisions),
                    "recall_media": np.mean(recalls),
                    "f1_media": np.mean(f1s),
                    "configuração": nome_config,
                })

    return pd.DataFrame(resumo_list)

 
def executar_experimentos_texto_com_parametros(
    textos,
    y,
    funcao_treino,
    vetorizadores_config
):
    
    def criar_vetorizador(tecnica, params):
        if tecnica == "Bag of Words":
            return CountVectorizer(**params)
        elif tecnica == "TF-IDF":
            return TfidfVectorizer(**params)
        else:
            raise ValueError(f"Técnica desconhecida: {tecnica}")

    resultados_gerais = {}

    for tecnica, lista_params in vetorizadores_config.items():
        resultados_gerais[tecnica] = {}

        for params in lista_params:
            nome_config = f"{tecnica} | {params}"

            vectorizer = criar_vetorizador(tecnica, params)

            X = vectorizer.fit_transform(textos.astype(str))
            feature_names = vectorizer.get_feature_names_out()

            resultados = funcao_treino(X, y, feature_names)

            resultados_gerais[tecnica][nome_config] = resultados

    return resultados_gerais

