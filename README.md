## 📂 Estrutura do Projeto

Este projeto segue uma estrutura organizada para garantir a reprodutibilidade e facilitar a colaboração.

* **`/data`**: Contém todos os dados do projeto, separados por estágio.
    * **`/data/raw`**: Dados brutos e imutáveis. **Os arquivos desta pasta nunca devem ser editados.**
    * **`/data/interim`**: Arquivos de dados intermediários e temporários.
    * **`/data/processed`**: Os conjuntos de dados finais, limpos e prontos para análise e modelagem.
    * **`/data/external`**: Dados de fontes externas ou de terceiros.

* **`/notebooks`**: Notebooks Jupyter (`.ipynb`) usados para exploração, prototipagem de modelos e análise de dados.

* **`/src`**: Código fonte (`.py`) reutilizável do projeto. Funções de limpeza, pipelines de processamento e classes de modelos devem ser armazenadas aqui para serem importadas pelos notebooks.

* **`/reports`**: Resultados finais e entregáveis do projeto.
    * **`/reports/figures`**: Gráficos, plots e visualizações geradas pela análise.

* **`/references`**: Documentação de apoio, artigos, manuais e qualquer material de referência (ex: PDFs, descrições de datasets).