import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import datetime
from glob import glob
from scipy.stats import pearsonr,binned_statistic_2d
from pathlib import Path
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

import re
from unidecode import unidecode



class elasticidade:
    def __init__(self) -> None:
        pass
    def _apply_date_filter(self, df, col_data, start_date, end_date):
        """
        Aplica filtro de data se start_date ou end_date forem fornecidos.
        """
        if start_date or end_date:
            # Garante que a coluna de data é datetime
            df[col_data] = pd.to_datetime(df[col_data])
            
            if start_date:
                data_inicio = pd.to_datetime(start_date)
                df = df[df[col_data] >= data_inicio].copy()
                
            if end_date:
                data_fim = pd.to_datetime(end_date)
                df = df[df[col_data] <= data_fim].copy()
                
            print(f"Base filtrada por data. {len(df)} registros restantes.")
            
        return df
        
    def prepara_base_elasticidade(self, df, cols_group, col_data, col_receita, col_volume, start_date=None, end_date=None):
        """
        Prepara a base de dados para cálculo de elasticidade.
        Parâmetros:
        - df: DataFrame original.
        - cols_group: Lista de colunas para agrupamento inicial.
        - col_data: Coluna de data.
        - col_receita: Coluna de receita.
        - col_volume: Coluna de volume.
        - start_date: Data inicial para filtro (opcional). YYYY-MM-DD
        - end_date: Data final para filtro (opcional). YYYY-MM-DD
        Retorna:
        - df_final: DataFrame preparado com listas históricas de preço e volume.
        """
        
        df_agg = df.groupby(cols_group).agg({
            col_receita: 'sum', 
            col_volume: 'sum'
        }) 

        df = df_agg.reset_index()
        col_preco = 'preco'
        df[col_preco] = df[col_receita] / df[col_volume]
        
        df = self._apply_date_filter(df, col_data, start_date, end_date)
        
        cols_time_series_key = [col for col in cols_group if col != col_data]
        
        df_final = df.dropna().groupby(cols_time_series_key).agg({
            col_volume: lambda x: list(x),
            col_preco: lambda x: list(x),
            col_data: lambda x: list(x),
        }).reset_index()
        
        return df_final

    def calculo_elasticidade_discreta(self,df,
                            col_resp,
                            col_variacao_percentual):

        try:
            X = pd.DataFrame([df[col_variacao_percentual],df[col_resp]]).T

            X.columns = [col_variacao_percentual, col_resp]
            X['qt'] = pd.qcut(X[col_variacao_percentual], 4, labels=False)
            X = X.groupby('qt').mean()
            Y = X[col_resp] # Volume
            Y = np.array(Y)

            X = X.drop(columns=[col_resp]) # Preço

            relativ_ref = X.iloc[-1][X.columns].copy()
            p0 = relativ_ref.values.copy()
            p0 = np.concatenate(([1],p0),axis=0)

            relativ_ref[col_variacao_percentual] = relativ_ref[col_variacao_percentual]*1.01
            p1 = relativ_ref.values.copy()
            p1 = np.concatenate(([1],p1),axis=0)


            X = np.array(X[col_variacao_percentual])
            X = sm.add_constant(X, has_constant='add')



            model = sm.OLS(Y, X)
    
                
            result = model.fit()
            
            rsquared = result.rsquared
            coef_pvalue = result.f_pvalue
            r2 = result.rsquared
            if np.isinf(r2) or np.isnan(r2):
                r2 = None
            
            p_val = result.f_pvalue
            if np.isinf(p_val) or np.isnan(p_val):
                p_val = None
            

            regr_a = result.predict(p0)[0]
            
            regr_b = result.predict(p1)[0]

            
            elasticidade = regr_b / regr_a -1
            print('Elasticidade discreta: ', elasticidade, 
                  df.get('cod_produto', df.get('sku_produto', 'ID Desconhecido')))
            return elasticidade,rsquared, coef_pvalue,X, Y
        
        except Exception as e:
            prod_id = df.get('produto', df.get('cod_produto', df.get('sku_produto', 'Desconhecido')))
            print(f'ERRO no produto {prod_id}: {e}')
            return None
    def cria_features_tempo(self,df,col_data):
        df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
        
        df['trimestre'] = df[col_data].dt.quarter
        df['mes'] = df[col_data].dt.month
        df['ano'] = df[col_data].dt.year
        df['dia_semana'] = df[col_data].dt.weekday


        return df

    def calculo_elasticidade_classica(self,
                            df,
                            colunas_x,
                            col_data,
                            col_resp,
                            col_variacao_percentual):
        

        """
        Calcula a elasticidade preço-volume usando regressão linear clássica.
        Parâmetros:
        - df: DataFrame contendo os dados.
        - colunas_x: Lista de colunas independentes (features).
        - col_data: Coluna de data para ordenação e criação de features temporais.
        - col_resp: Coluna dependente,  volume.
        - col_variacao_percentual: Coluna de variação percentual do preço.
        Retorna:
        - elasticidade: Valor da elasticidade calculada.
        - rsquared: Coeficiente de determinação da regressão.
        - coef_pvalue: Valor-p do teste F da regressão.
        - X: Matriz de variáveis independentes usadas na regressão.
        - Y: Vetor da variável dependente usada na regressão.
        """

        try:
            X = pd.DataFrame()

            for col in colunas_x:
                X[col] = df[col]

            X[col_data] = df[col_data]
            
            X = self.cria_features_tempo(X,col_data)
            X.sort_values(by=[col_data],inplace=True)

            X.drop(columns=[col_data],inplace=True)
            
            # ano_corte = 2022
            # mes_corte = 1
    
            # filtro = (X['ano'] > ano_corte) | ((X['ano'] == ano_corte) & (X['mes'] >= mes_corte))
            
            # X = X[filtro]
            X = X[X['preco']>0]  # Remover preços negativos ou zero
            relativ_ref = X.iloc[-1][X.columns].copy()
            p0 = relativ_ref.values.copy()
            p0 = np.concatenate(([1],p0),axis=0)

            relativ_ref[col_variacao_percentual] = relativ_ref[col_variacao_percentual]*1.01
            p1 = relativ_ref.values.copy()
            p1 = np.concatenate(([1],p1),axis=0)
            
            Y_full = np.array(df[col_resp])
            Y = Y_full[X.index]

            X = np.array(X)
            X = sm.add_constant(X, has_constant='add')

            # print(Y)
            model = sm.OLS(Y, X)
    
                
            result = model.fit()
            
            rsquared = result.rsquared
            coef_pvalue = result.f_pvalue
            r2 = result.rsquared
            if np.isinf(r2) or np.isnan(r2):
                r2 = None
            
            p_val = result.f_pvalue
            if np.isinf(p_val) or np.isnan(p_val):
                p_val = None
            
            
            regr_a = result.predict(p0)[0]
            
            regr_b = result.predict(p1)[0]

            elasticidade = regr_b / regr_a -1

            print('Elasticidade clássica: ', elasticidade, df.get('cod_produto', df.get('sku_produto', 'ID Desconhecido')))
            return elasticidade,rsquared, coef_pvalue,X, Y

        except Exception as e:
            # Tenta pegar 'produto', 'cod_produto' ou usa 'Desconhecido'
            prod_id = df.get('produto', df.get('cod_produto', 'Desconhecido'))
            print(f'ERRO no produto {prod_id}: {e}')
            return None
        
    def calculo_elasticidade(self,
                            df,
                            colunas_x,
                            col_data,
                            col_resp,
                            col_variacao_percentual,
                            tam_amostra,
                            threshold_metodo=30):
        
        if tam_amostra < 4:
            return None
        
        elif tam_amostra < threshold_metodo:
            return self.calculo_elasticidade_discreta(df, col_resp, col_variacao_percentual)
        else:
            return self.calculo_elasticidade_classica(df, colunas_x, col_data, col_resp, col_variacao_percentual)

    def executar(self, 
                    df_input: pd.DataFrame, 
                    colunas_granularidade: list,
                    col_data: str, 
                    col_receita: str, 
                    col_volume: str,
                    output_path_parquet: str = None,
                    output_path_excel: str = None,
                    threshold_amostra: int = 30,
                    start_date: str = None,
                    end_date: str = None):
            """
            Executa o pipeline completo de elasticidade com parâmetros definidos pelo usuário.
            
            Parâmetros:
            - df_input: DataFrame com os dados brutos (transações).
            - colunas_granularidade: Lista de colunas para agrupar (ex: ['cod_produto', 'regiao']).
            - col_data: Nome da coluna de data.
            - col_receita: Nome da coluna de valor monetário.
            - col_volume: Nome da coluna de quantidade.
            - output_path_parquet: Caminho para salvar o resultado em Parquet (opcional).
            - output_path_excel: Caminho para salvar o resultado em Excel (opcional).
            - threshold_amostra: Limite para usar o método clássico (padrão 30).
            - start_date/end_date: Filtros opcionais de data.
            """
            
            print(f"Iniciando cálculo de elasticidade. Granularidade: {colunas_granularidade}")

            
            elasticidade_df = self.prepara_base_elasticidade(
                df=df_input,
                cols_group=colunas_granularidade + [col_data],
                col_data=col_data,
                col_receita=col_receita, 
                col_volume=col_volume,   
                start_date=start_date,
                end_date=end_date
            )
            col_preco_calc = 'preco' 
            
            elasticidade_df['tam_amostral'] = elasticidade_df[col_data].apply(len)
            elasticidade_df['precos_unicos'] = elasticidade_df[col_preco_calc].apply(lambda x: len(set(x)))

            print(f"Itens processados: {len(elasticidade_df)}")
            print(f"Itens sem variação de preço: {len(elasticidade_df[elasticidade_df['precos_unicos'] <= 1])}")

            # 3. Cálculo da Elasticidade (Iteração)
            elasticidade_df['resultados_raw'] = elasticidade_df.apply(
                lambda row: self.calculo_elasticidade(
                    df=row,
                    colunas_x=[col_preco_calc],
                    col_data=col_data,
                    col_resp=col_volume, # A coluna de volume (lista)
                    col_variacao_percentual=col_preco_calc,
                    tam_amostra=row['tam_amostral'],
                    threshold_metodo=threshold_amostra # <--- Parâmetro do usuário
                ),
                axis=1
            )

            # 4. Desempacotamento dos Resultados
            # O retorno é (elasticidade, r2, pvalue, X, Y) ou None
            elasticidade_df['elasticidade'] = elasticidade_df['resultados_raw'].apply(lambda x: x[0] if x else 0)
            elasticidade_df['r2'] = elasticidade_df['resultados_raw'].apply(lambda x: x[1] if x else None)
            elasticidade_df['p_value'] = elasticidade_df['resultados_raw'].apply(lambda x: x[2] if x else None)
            
            # Remove colunas temporárias pesadas se for salvar
            elasticidade_df.drop(columns=['resultados_raw'], inplace=True)

            # 5. Salvamento (Opcional)
            if output_path_parquet:
                elasticidade_df.to_parquet(output_path_parquet)
                print(f"Salvo em Parquet: {output_path_parquet}")
                
            if output_path_excel:
                elasticidade_df.to_excel(output_path_excel)
                print(f"Salvo em Excel: {output_path_excel}")
            
            return elasticidade_df

if __name__ == "__main__":
    elasticidade = elasticidade()
    elasticidade.executar()
    