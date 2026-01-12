import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def analise_metricas(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def kfold_gridsearch_classificacao(X, y, features_names):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features_names)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    modelos = {
        "KNN": {
            "model": KNeighborsClassifier,
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "metric": ["euclidean", "manhattan"]
            }
        },
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
                "penalty": ["l2"],
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

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {nome: [] for nome in modelos}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        print(f"\n===== Fold {fold} =====")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

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
                    average="macro",
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
                "fold": fold + 1,
                "modelo": final_model,
                "scaler": scaler_final,
                "best_params": best_params,
                "y_test": y_test,
                "y_pred": y_test_pred,
                "metricas": analise_metricas(y_test, y_test_pred)
            })

    return resultados


def pegar_dados_modelo_final(resultados, nome_modelo_final):
    folds = resultados[nome_modelo_final]
    
    y_true_total = np.concatenate([fold_info['y_test'] for fold_info in folds])
    y_pred_total = np.concatenate([fold_info['y_pred'] for fold_info in folds])
    
    modelo_final = folds[0]['modelo']
    scaler_final = folds[0].get('scaler', None)
    
    return y_true_total, y_pred_total, modelo_final, scaler_final


def treinar_modelo_SHAP(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Prever e avaliar
    y_pred = modelo.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))

