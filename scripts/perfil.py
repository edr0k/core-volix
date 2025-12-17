import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import re
from unidecode import unidecode
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
import pickle
import networkx as nx
from pyvis.network import Network

class perfil:
    def __init__(self) -> None:
        pass

    def ajusta_nome_colunas (self,coluna):
        """
        Função para ajustar nomes de colunas, removendo caracteres especiais, espaços e acentuação.
        
        Parâmetros:
        coluna (str): Nome da coluna a ser ajustado.
        """

        coluna = coluna.lower().strip()
        dAjuste = {"%":"pct",
                " ":"_"}
        
        for de, para in dAjuste.items():
            coluna = coluna.replace(de, para)
            coluna=re.sub(r'[^\w ]',"",coluna)
            coluna = unidecode(coluna)
        return coluna

    def modelagem(self,X,scaled_features,n_clusters,nome_modelo,colunas_categoricas):
        """
        Função para realizar a modelagem utilizando KPrototypes.

        Parâmetros:
        X (DataFrame): Dados originais.
        scaled_features (array): Dados escalonados.
        n_clusters (int): Número de clusters.
        nome_modelo (str): Nome do modelo.
        colunas_categoricas (list): Índices das colunas categóricas.
        """
        
        X_sc = pd.DataFrame(scaled_features, columns=X.columns)
        kmeans = KPrototypes(n_clusters=n_clusters,max_iter=20).fit(X_sc,categorical=colunas_categoricas)

        resultados =X_sc.copy()
        resultados['cluster'] = kmeans.predict(X_sc,categorical=colunas_categoricas)
        resultados.index = X.index

        cluster_labels = list(resultados['cluster'].unique())
        silhouette_vals = silhouette_samples(scaled_features, resultados['cluster'])

        resultados['silhueta'] = silhouette_vals
        silhueta = resultados.groupby('cluster').agg({'silhueta': 'mean'})
        print(n_clusters,silhueta.mean().values[0])

        # with open(fr"../modelos/teste_{nome_modelo}.pkl", "wb") as f:
        #     pickle.dump(kmeans, f)

        return resultados
    def gera_info_cluster(self,dfg,cluster,save_file=False):

        net_graph = nx.Graph()
        net_graph.clear()
        for index, row in dfg[dfg.cluster==cluster].iterrows():
            no_de = row['cliente_grupo']
            no_para = row['familia_repasse_1']
            peso = row['volume_t']


            net_graph.add_node(no_de)
            net_graph.add_node(no_para)
            net_graph.add_edge(no_de, no_para, weight=peso)

        
        nt = Network()
        nt.from_nx(net_graph)
        if save_file:
            nt.show(f'cluster_{cluster}.html',notebook=False)

        centr_grau = nx.degree_centrality(net_graph) 
        centr_prox =  nx.closeness_centrality(net_graph) 
        centr_betw = nx.betweenness_centrality(net_graph)

        return net_graph,centr_grau,centr_prox,centr_betw

    def gerar_tabelas(self,resultados, base, base_perfil):

        N_CLUSTERS = 7 
        SCRIPT_DIR = Path(__file__).resolve().parent 
        PROJECT_ROOT = SCRIPT_DIR.parent.parent 
        DATA_DIR = PROJECT_ROOT / 'data'
        local_resultados = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_antiga.parquet'
        local_metricas_parquet = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_metricas.parquet'
        local_metricas_excel = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_metricas.xlsx'
        local_pedido_sugerido_parquet = DATA_DIR / f'dados_processados/pedido_sugerido_n{N_CLUSTERS}.parquet'
        local_pedido_sugerido_excel = DATA_DIR / f'dados_processados/pedido_sugerido_n{N_CLUSTERS}.xlsx'

        df2 = pd.merge(base, resultados.cluster, on = 'cliente_grupo')
        df2.cluster.unique()
        df3 = pd.merge(base_perfil, resultados.cluster, on = 'cliente_grupo').reset_index()
        df3['perfil'] = df3.cluster.replace([2, 3, 4, 5, 6, 0, 1], ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

        df3 = df3[[
            'cliente_grupo',
            'cluster',
            'perfil'
        ]]

        clusters = df3.copy()

        clusters.cluster = clusters.cluster.astype(str)

        d_clusters = dict(zip(clusters['cliente_grupo'], clusters['cluster']))

        dfg = base.copy()
        dfg['cluster'] = dfg['cliente_grupo'].map(d_clusters)


        dfg = dfg.groupby(['cluster','familia_repasse_1','cliente_grupo']).agg({'volume_t':'sum'}).reset_index()
        dfg2 = dfg.groupby(['cluster','familia_repasse_1']).agg({'volume_t':'sum'}).reset_index()
        dfg = dfg.merge(dfg2, on=['cluster','familia_repasse_1'],suffixes=('','_total'))
        dfg['cluster'] = dfg['cluster'].astype(str)

        clusters = list(dfg.cluster.unique())
        n_clusters = len(clusters)

        # for cluster in clusters:
        #     cluster = str(cluster)
        #     self.gera_grafico_cluster(dfg,cluster,save_file=False)

        df_metricas = pd.DataFrame()
        for cluster in list(range(0,N_CLUSTERS+2)):
            cluster = str(cluster)

            net_graph,centr_grau,centr_prox,centr_betw = self.gera_info_cluster(dfg,cluster, False)

            df_metricas_i = pd.DataFrame()
            for nome, centr in [['grau_centralidade',centr_grau],['proximidade', centr_prox],['intermediação',centr_betw]]:
                metricas = pd.DataFrame.from_dict(centr,orient='index',columns=[nome])
                metricas = metricas[metricas.index.isin(dfg.familia_repasse_1.unique())].copy()
                metricas = pd.DataFrame(metricas.stack())
                metricas.columns = [cluster]
                df_metricas_i = pd.concat([df_metricas_i,metricas])

            df_metricas = pd.concat([df_metricas,df_metricas_i],axis=1)

        #df_metricas = df_metricas.stack().unstack(level=0)
        df_metricas = df_metricas.stack().unstack(level=1).reset_index().rename(columns={'level_0':'familia_repasse_1',
                                                                                        'level_1':'cluster'})	

        std_centr = df_metricas.groupby('cluster')[['grau_centralidade']].std()
        std_centr.rename(columns={'grau_centralidade':'std_grau_centralidade'}, inplace=True)
        std_centr.reset_index(inplace=True)
        df_metricas = df_metricas.merge(std_centr, on = ['cluster'])
        df_metricas
        df_metricas.grau_centralidade.quantile(.4)
        ref_centralidade = df_metricas.grau_centralidade.quantile(.4)
        ref_intermediacao = df_metricas['intermediação'].quantile(.75)




        df_metricas['classificacao'] = df_metricas.apply(lambda x: self.classificacao(x['grau_centralidade'],
                                                                                x['intermediação'],
                                                                                x['std_grau_centralidade'],
                                                                                ref_centralidade,
                                                                                ref_intermediacao),axis=1)


        mix = base.copy()
        mix['cluster'] = mix['cliente_grupo'].map(d_clusters)
        mix = mix.groupby(['cluster','familia_repasse_1']).agg({'volume_t':'sum'}).reset_index()

        mix_t = base.copy()
        mix_t['cluster'] = mix_t['cliente_grupo'].map(d_clusters)
        mix_t = mix_t.groupby(['cluster']).agg({'volume_t':'sum'}).reset_index()

        mix = mix.merge(mix_t, on=['cluster'],suffixes=('','_total'))
        mix['peso_mix'] = mix.volume_t / mix.volume_t_total
        mix.drop(columns=['volume_t','volume_t_total'],inplace=True)


        df_metricas.cluster.dtypes, mix.cluster.dtypes
        df_metricas = df_metricas.merge(mix, on=['cluster','familia_repasse_1'], how='outer')
        df_metricas.to_parquet(local_metricas_parquet,index=False)
        df_metricas.to_excel(local_metricas_excel,index=False)

        ps = base.copy()
        ps['cluster'] = ps['cliente_grupo'].map(d_clusters)
        ps = ps.merge(mix, on=['cluster','familia_repasse_1'], how='outer').fillna(0)
        ps.cluster = ps.cluster.astype(float)
        df_metricas.cluster = df_metricas.cluster.astype(float)
        ps = ps.merge(df_metricas[['cluster','familia_repasse_1', 'classificacao']], on=['cluster','familia_repasse_1'])
        ps = ps[ps['regiao_2']!='regiao_2']
        ps.nota_fiscal.fillna('-', inplace=True)
        ps.nota_fiscal = ps.nota_fiscal.astype(str)
        ps.cliente= ps.cliente.astype(str)
        ps.cliente_grupo= ps.cliente_grupo.astype(str)
        ps.gerente_comercial= ps.gerente_comercial.astype(str)
        ps.gerente_contas= ps.gerente_contas.astype(str)
        ps.tipologia= ps.tipologia.astype(str)
        ps.grupo_tipologico= ps.grupo_tipologico.astype(str)
        ps.to_parquet(local_pedido_sugerido_parquet, index = True)
        ps.reset_index().to_excel(local_pedido_sugerido_excel, index = False)

        print(f"Tabelas geradas!" )
        return ps

    def classificacao(self,grau_centralidade,intermediacao,str_centr,ref_centralidade,ref_intermediacao):
        """
        Função de classificação de prioridade.

        """

        if  str_centr< 0.1:
            return 'Prioritário'
        elif grau_centralidade >= ref_centralidade and intermediacao >= ref_intermediacao:
            return 'Prioritário - Indispensável'
        elif grau_centralidade >= ref_centralidade:
            return 'Prioritário'
        else:
            return 'Potencial'
        
    def executar(self):
        """
        Função principal para executar o processo de perfilagem.
        """

        N_CLUSTERS = 7 
        SCRIPT_DIR = Path(__file__).resolve().parent 
        PROJECT_ROOT = SCRIPT_DIR.parent.parent 
        DATA_DIR = PROJECT_ROOT / 'data'
        # local_resultados = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_antiga.parquet'
        # local_metricas_parquet = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_metricas.parquet'
        # local_metricas_excel = DATA_DIR / f'dados_processados/clusterizacao_n{N_CLUSTERS}_metricas.xlsx'
        # local_pedido_sugerido_parquet = DATA_DIR / f'dados_processados/pedido_sugerido_n{N_CLUSTERS}.parquet'
        # local_pedido_sugerido_excel = DATA_DIR / f'dados_processados/pedido_sugerido_n{N_CLUSTERS}.xlsx'
        base_pedido_sugerido = DATA_DIR / 'dados_processados/base_dados_pedido_sugerido.parquet'
        local_base_perfil = DATA_DIR / 'auxiliar/base_perfil_cliente_grupo.xlsx'
        local_base_ajuste_perfil = DATA_DIR / 'auxiliar/ajustes_novas_areas_CONFIRMAR_VIVIX.xlsx'
        nome_modelo = f'kmeans_{N_CLUSTERS}_1'

        base = pd.read_parquet(base_pedido_sugerido)

        base.dtfaturamento = pd.to_datetime(base.dtfaturamento)

        filtro = (base.dtfaturamento >= base.dtfaturamento.max()-relativedelta(months=24))

        base = base[filtro].copy()
        filtro = (base.grupo_tipologico == 'Transformadores')

        df = base[filtro].copy()


        X_sku = df.groupby(['cliente_grupo']).agg({'material_sku': pd.Series.nunique})
        X_sku.rename(columns = {'material_sku': 'numero_de_skus'}, inplace = True)

        X_vol = df.groupby([
            'cliente_grupo',
            ]).agg({'volume_t': 'sum'})

        X_peso_esp = df.groupby([
            'cliente_grupo',
            'familia_repasse_1'
            ]).agg({'receita_liquida': 'sum'}).unstack('familia_repasse_1').fillna(0)

        X_peso_esp.columns = X_peso_esp.columns.droplevel(0)
        X_peso_esp['total'] = X_peso_esp.sum(axis = 1)
        X_peso_esp['produtos_especiais'] = X_peso_esp[['Laminado', 'PSolar', 'Espelho', 'Pintado']].sum(axis = 1)
        X_peso_esp['peso_especial'] = X_peso_esp.produtos_especiais / X_peso_esp.total
        X_peso_esp = X_peso_esp[['peso_especial']].copy()

        base_perfil = pd.read_excel(local_base_perfil)
        ajuste = pd.read_excel(local_base_ajuste_perfil)
        for item in ajuste.values:
            indice = base_perfil[base_perfil.cliente_grupo==item[0]].index[0]
            base_perfil.iloc[indice,0] = item[1]


        base_perfil.columns = [self.ajusta_nome_colunas(x) for x in base_perfil.columns]
        base_perfil['cliente_grupo'] = base_perfil.cliente_grupo + '|' + base_perfil.id_gerente_contas
        base_perfil = base_perfil.drop(columns = 'id_gerente_contas')
        base_perfil = base_perfil.replace('x', 1)

        cols_perfil = [
            'perfil_temp',
            'perfil_dist',
            'perfil_rota',
            'perfil_balcao',
            'perfil_loja',
            'perfil_obras',
            'perfil_instalacao',
            'perfil_esquadrias',
            'perfil_moveleiro',
            'perfil_outros'	
            ]

        base_perfil = base_perfil.fillna(0)
        base_perfil.groupby('cliente_grupo').nunique().max()[cols_perfil].nunique() == 1
        base_perfil.drop_duplicates(subset = ['cliente_grupo'] + cols_perfil, inplace = True)
        base_perfil.set_index('cliente_grupo', inplace = True)
        base_perfil = base_perfil[cols_perfil].copy()
        base_perfil.fillna(0, inplace = True)
        base_perfil = base_perfil.astype(int)
        X = pd.concat([
            X_sku,
            X_vol,
            X_peso_esp,
            base_perfil
            ], axis = 1)
        X.dropna(inplace = True)

        cols_numericas = ['numero_de_skus','volume_t','peso_especial']

        scaler = StandardScaler()
        scaled_features = X.copy()
        scaled_features[cols_numericas] = scaler.fit_transform(X[cols_numericas])
        X_perfil = X[X[cols_perfil].sum(axis = 1) > 0].copy()
        X_perfil = X_perfil[cols_numericas + cols_perfil].copy()    

        X_tudo = scaled_features
        X_tudo['cluster'] = 99

        # X_tudo['cluster'] = X_tudo['cluster'].astype(object)

        # perfil A
        for i in range(len(X_tudo)):
            # Use .iloc[i] para pegar a linha pela POSIÇÃO
            row = X_tudo.iloc[i]
            if (row['perfil_temp'] == 1) and (row['perfil_dist'] == 1) and (row['perfil_balcao'] == 1) and (row['perfil_loja'] == 1):
                X_tudo.iloc[i, -1] = 2

        # perfil B
        for i in range(len(X_tudo)):
            row = X_tudo.iloc[i]
            if (row['perfil_temp'] == 0) and (row['perfil_obras'] == 1):
                X_tudo.iloc[i, -1] = 3

        # perfil C
        for i in range(len(X_tudo)):
            row = X_tudo.iloc[i]
            if (row['perfil_balcao'] == 0) and (row['perfil_obras'] == 0) and (row['perfil_rota'] == 0):
                X_tudo.iloc[i, -1] = 4

        # perfil D
        for i in range(len(X_tudo)):
            row = X_tudo.iloc[i]
            if (row['perfil_dist'] == 1) and (row['perfil_obras'] == 0 ):
                X_tudo.iloc[i, -1] = 5

        # perfil E
        for i in range(len(X_tudo)):
            row = X_tudo.iloc[i]
            if (row['perfil_dist'] == 0) and (row['perfil_loja'] == 1):
                X_tudo.iloc[i, -1] = 6

        resto = X_tudo[X_tudo.cluster == 99].drop(columns = 'cluster')
        X_tudo = X_tudo[X_tudo.cluster != 99].copy()
        

        print('Tamanho X_tudo: ', len(X_tudo), 'Tamanho resto: ', len(resto))
        resultados = pd.DataFrame()

        X_mod = resto.copy()
        scaled_features = X_mod.copy() # dados já passaram por um fit_transform antes
        # scaled_features[cols_numericas] = scaler.fit_transform(X_mod[cols_numericas])
        #if remodelar:
        resultados_i = self.modelagem(X_mod ,scaled_features , 2, nome_modelo, colunas_categoricas = list(range(len(cols_numericas), X_mod.shape[1])))
        #else:
            #resultados_i = f.carregar_modelo(X_mod, scaled_features, 7, nome_modelo, colunas_categoricas = list(range(len(cols_numericas), X_mod.shape[1])))

        #resultados_i['cluster']  = resultados_i['cluster'] + dic_modelos['perfil'][3]
        resultados = pd.concat([resultados, resultados_i], axis = 0)

        nao_clusterizados = set(base.cliente_grupo) - set(resultados.index)
        nao_clusterizados = base[base.cliente_grupo.isin(nao_clusterizados)][['cliente_grupo', 'grupo_tipologico']].drop_duplicates()
        nao_clusterizados['cluster'] = nao_clusterizados.apply(lambda x: resultados.cluster.max() + 1 if x.grupo_tipologico == 'Distribuidores' else  resultados.cluster.max() + 2, axis = 1)
        nao_clusterizados = nao_clusterizados.set_index('cliente_grupo')[['cluster']]
        resultados = pd.concat([resultados, nao_clusterizados])
        scaled_columns = scaled_features.columns

        cluster = resultados[[
            'cluster',
            #'silhueta'
            ]]

        resto = resto.merge(cluster, on = 'cliente_grupo').copy()

        tudin = pd.concat([X_tudo, resto])

        minimum_num = 1 # the minumum number of the cluster size
        ok_labels = ['5']
        labels = tudin['cluster'].unique().tolist()

        sns.scatterplot(data = tudin, x = 'numero_de_skus', y = 'volume_t', hue = 'cluster', palette = 'tab10')
        plt.title('Clusters KMeans PCA - Perfil')
        # plt.show()


        ps = self.gerar_tabelas(tudin, base, base_perfil)
        for i in range(0,7):
            print('Cluster ', i, ': ', len(ps[ps['cluster']==i]))
if __name__ == "__main__":

    perfil = perfil()
    perfil.executar()
    
  


        