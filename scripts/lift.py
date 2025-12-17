import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import os
from pathlib import Path

pd.set_option('display.max_columns', None)

class Lift:
    def __init__(self, n_clusters=7):
        """
        Inicializa a classe de análise de Lift, definindo parâmetros e caminhos.
        """
        self.N_CLUSTERS = n_clusters
        self.DATA_DIR = None
        self.lift_results_dir = None
        
        # Paths de arquivos
        self.base_pedido_sugerido_path = None
        self.local_pedido_sugerido_path = None
        
        # Parâmetros da análise
        self.coluna_cliente = 'cliente_grupo'
        self.coluna_item = 'produto'
        self.coluna_peso = 'receita_liquida'
        self.coluna_cluster = 'cluster'
        
        # DataFrames
        self.df = None
        self.produtos_top95 = None

    def _setup_paths(self):
        """
        Configura os caminhos de diretório e arquivos.
        Cria o diretório de saída se ele não existir.
        """
        try:
            SCRIPT_DIR = Path(__file__).resolve().parent 
        except NameError:
            # Fallback se rodando em um notebook interativo
            SCRIPT_DIR = Path.cwd() 

        PROJECT_ROOT = SCRIPT_DIR.parent.parent 
        self.DATA_DIR = PROJECT_ROOT / 'data'
        
        # Paths de entrada
        self.base_pedido_sugerido_path = self.DATA_DIR / 'dados_processados' / 'base_dados_pedido_sugerido.parquet'
        self.local_pedido_sugerido_path = self.DATA_DIR / f'dados_processados/pedido_sugerido_n{self.N_CLUSTERS}.parquet'
        
        # Path de saída
        self.lift_results_dir = self.DATA_DIR / 'dados_processados/lift_regioes'
        
        # Cria o diretório de saída
        os.makedirs(self.lift_results_dir, exist_ok=True)
        print(f"Diretório de resultados: {self.lift_results_dir}")

    def _load_and_merge_data(self):
        """
        Carrega os dados base e os dados de cluster, 
        e os mescla para criar o DataFrame principal 'self.df'.
        """
        print("Carregando e mesclando dados...")
        df_base = pd.read_parquet(self.base_pedido_sugerido_path)
        df_pedido_sugerido = pd.read_parquet(self.local_pedido_sugerido_path)
        
        df_clusters = df_pedido_sugerido[['cliente_cod', 'cluster']].drop_duplicates(subset=['cliente_cod'])
        
        self.df = df_base.merge(df_clusters, on='cliente_cod')
        self.df = self.df[self.df.produto != 'produto'].copy() # Limpeza de dados
        print("Dados carregados e mesclados.")

    def _calculate_product_filter(self):
        """
        Calcula a "Curva ABC" (Top 95%) dos produtos com base na receita
        e armazena a lista em 'self.produtos_top95'.
        """
        print("Calculando filtro de produtos (Top 95%)...")
        produtos_agg = self.df.groupby(self.coluna_item)[[self.coluna_peso]].sum().sort_values(by=self.coluna_peso, ascending=False)
        produtos_agg['peso_cum'] = produtos_agg[self.coluna_peso].cumsum()
        total_revenue = produtos_agg[self.coluna_peso].sum()
        produtos_agg['peso_cum'] = (produtos_agg[self.coluna_peso] / total_revenue) * 100
        
        self.produtos_top95 = list(produtos_agg[produtos_agg['peso_cum'] <= 95].index.values)
        print(f"Análise focada em {len(self.produtos_top95)} produtos principais.")

    def _get_cluster_parameters(self, regiao, cluster):
        """
        Retorna os parâmetros de filtro e Apriori para um clustero específico.
        Trata o 'cluster 2 / Nordeste' como um caso especial.
        """
        if (cluster == 2) and (regiao == "Nordeste"):
            # Parâmetros RESTRITOS para o cluster gigante
            return {
                'use_product_filter': False, # Mudar para True caso queira usar o filtro
                'min_support': 0.10,        # Aumenta o suporte
                'max_len': 3                # Limita o tamanho das combinações
            }
        else:
            # Parâmetros NORMAIS para outros
            return {
                'use_product_filter': False, # Não usa o filtro
                'min_support': 0.05,
                'max_len': None
            }

    def _clean_rules(self, rules):
        """
        Converte colunas de frozenset para list e calcula tamanhos.
        """
        rules['antecedents'] = rules.antecedents.apply(lambda x: list(x))
        rules['consequents'] = rules.consequents.apply(lambda x: list(x))
        rules['tam_consequents'] = rules.consequents.str.len()
        rules['tam_antecedents'] = rules.antecedents.str.len()
        return rules

    def _process_cluster(self, regiao, cluster):
        """
        Executa o pipeline completo (filtro, crosstab, apriori, rules) 
        para um único cluster de região/cluster.
        """
        # 1. Obter parâmetros
        params = self._get_cluster_parameters(regiao, cluster)
        
        # 2. Filtrar DataFrame
        filters = [
            (self.df['regiao_2'] == regiao),
            (self.df[self.coluna_cluster] == cluster)
        ]
        
        # Aplica o filtro de produtos se especificado (para o cluster 2)
        if params['use_product_filter']:
            filters.append(self.df[self.coluna_item].isin(self.produtos_top95))

        df_filtro = self.df[np.logical_and.reduce(filters)].copy()
        
        if df_filtro.empty:
            return 

        # 3. Criar Pedidos e Crosstab
        df_filtro['pedido'] = df_filtro[self.coluna_cliente] + df_filtro['anomes_fatura']
        
        if df_filtro['pedido'].nunique() == 0:
            return

        print(f"Rodando para a Regiao {regiao}, e Cluster {cluster}...")
        df_crosstab = pd.crosstab(df_filtro['pedido'], df_filtro[self.coluna_item])
        
        if df_crosstab.empty:
            return

        # 4. Converter para Booleano
        compras_cliente = df_crosstab > 0

        # 5. Rodar Apriori
        frequent_itemsets = apriori(
            compras_cliente, 
            min_support=params['min_support'], 
            use_colnames=True, 
            max_len=params['max_len']
        )
        
        if frequent_itemsets.empty:
            print(f"   ...Nenhum itemset frequente para Regiao {regiao}, Cluster {cluster}.")
            return

        # 6. Gerar Regras
        rules = association_rules(frequent_itemsets, metric="lift")
        
        if rules.empty:
            print(f"   ...Nenhuma regra encontrada para Regiao {regiao}, Cluster {cluster}.")
            return

        # 7. Limpar e Salvar
        rules = self._clean_rules(rules)
        
        caminho_resultado = self.lift_results_dir / f'analise_lift_{regiao}_{cluster}.parquet'
        rules.to_parquet(caminho_resultado)
        print(f"   ...Regiao {regiao}, e Cluster {cluster} com tamanho {len(rules)}")

    def _consolidate_rules(self):
        """
        Consolida os arquivos Parquet de regras salvos em um único
        arquivo (Parquet e Excel) por região.
        """
        print("\n--- Consolidando arquivos por região ---")
        regioes = [x for x in self.df.regiao_2.unique() if x != 'regiao_2']
        
        for regiao in regioes:
            all_rules = pd.DataFrame()
            for cluster in range(0, self.N_CLUSTERS):
                try:
                    path_rules_cluster = self.lift_results_dir / f'analise_lift_{regiao}_{cluster}.parquet'
                    rules_cluster = pd.read_parquet(path_rules_cluster)
                    rules_cluster['cluster'] = cluster
                    all_rules = pd.concat([all_rules, rules_cluster])
                    print(f"Abriu cluster {cluster} da região {regiao}")
                except FileNotFoundError:
                    print(f"Nenhum arquivo de regra para cluster {cluster}, região {regiao}")
                except Exception as e:
                    print(f"Falha em abrir/concatenar cluster {cluster}, região {regiao}: {e}")
            
            if not all_rules.empty:
                all_rules.to_parquet(self.DATA_DIR / f'dados_processados/analise_lift_{regiao}.parquet')
                all_rules.to_excel(self.DATA_DIR / f'dados_processados/analise_lift_{regiao}.xlsx')
                print(f"Arquivo consolidado para Regiao {regiao} salvo.")
            else:
                print(f"Nenhuma regra encontrada para Regiao {regiao}.")

    def executar(self):
        """
        Método principal para orquestrar e executar todo o pipeline de análise.
        """
        # 1. Configuração
        self._setup_paths()
        
        # 2. Carga e Preparação de Dados
        self._load_and_merge_data()
        self._calculate_product_filter()
        
        # 3. Processamento
        print("\n--- Iniciando geração de regras por cluster")
        regioes = [x for x in self.df.regiao_2.unique() if x != 'regiao_2']
        for cluster in range(0, self.N_CLUSTERS):
            for regiao in regioes:
                try:
                    # Processa cada cluster individualmente
                    self._process_cluster(regiao, cluster)
                except Exception as e:
                    print(f"ERRO FATAL ao processar Regiao {regiao}, Cluster {cluster}: {e}")
        
        # 4. Consolidação 
        self._consolidate_rules()
        
        print("\nAnálise de Lift concluída com sucesso.")

if __name__ == "__main__":
    analise = Lift(n_clusters=7)
    analise.executar()