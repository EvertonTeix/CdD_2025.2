import numpy as np
import pandas as pd

def separar_numericos_categoricos(dataframe):

    df_numerico = dataframe.select_dtypes(include=["int64", "float64"])
    df_categorico = dataframe.select_dtypes(include=["object"])

    return df_numerico, df_categorico


def tratar_categoricos_como_numericos(df_categorico):

    df_categorico_numerico = pd.get_dummies(
        df_categorico,
        drop_first=False,
        dtype=int
    )

    return df_categorico_numerico

def juntar_numericos_e_categoricos(df_numerico,df_categorico_numerico):

    dataframe_final = pd.concat(
        [df_numerico, df_categorico_numerico],
        axis=1
    )

    return dataframe_final


def remover_features_correlacionadas(dataframe,limiar=0.7,verbose=True):

    corr_matrix = dataframe.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    colunas_remover = []

    if verbose:
        print(f"Colunas com correlação >= {limiar}:\n")

    for col in upper.columns:
        correlacoes_altas = upper[col][upper[col] >= limiar]
        if not correlacoes_altas.empty:
            mais_correlacionada = correlacoes_altas.idxmax()
            valor = correlacoes_altas.max()

            if verbose:
                print(
                    f"- {col} está altamente correlacionada com "
                    f"{mais_correlacionada} (correlação = {valor:.2f})"
                )

            colunas_remover.append(col)

    dataframe_sem_correlacao = dataframe.drop(columns=colunas_remover)

    return dataframe_sem_correlacao, colunas_remover


def resultados_para_dataframe(resultados):
    linhas = []

    for nome_modelo, folds in resultados.items():
        for info in folds:
            linha = {
                "modelo": nome_modelo,
                "fold": info["fold"],
                "accuracy": info["metricas"]["accuracy"],
                "precision": info["metricas"]["precision"],
                "recall": info["metricas"]["recall"],
                "f1": info["metricas"]["f1"],
                "best_params": str(info["best_params"])
            }
            linhas.append(linha)

    return pd.DataFrame(linhas)

def tabela_metricas_medias(resultados):
    linhas = []

    for nome_modelo, folds in resultados.items():
        accs, precisions, recalls, f1s = [], [], [], []
        best_params_f1 = None
        melhor_f1 = -np.inf

        for fold_info in folds:
            accs.append(fold_info['metricas']['accuracy'])
            precisions.append(fold_info['metricas']['precision'])
            recalls.append(fold_info['metricas']['recall'])
            f1s.append(fold_info['metricas']['f1'])

            if fold_info['metricas']['f1'] > melhor_f1:
                melhor_f1 = fold_info['metricas']['f1']
                best_params_f1 = fold_info['best_params']

        linha = {
            'Modelo': nome_modelo,
            'Accuracy Média': np.mean(accs),
            'Precision Média': np.mean(precisions),
            'Recall Médio': np.mean(recalls),
            'F1-score Médio': np.mean(f1s),
            'Melhor Parâmetro': best_params_f1
        }
        linhas.append(linha)

    df_metricas = pd.DataFrame(linhas)
    df_metricas = df_metricas.sort_values(by='F1-score Médio', ascending=False).reset_index(drop=True)
    return df_metricas