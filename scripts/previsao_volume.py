import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prophet import Prophet
from pathlib import Path
import logging
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

class previsao_volume:
    def __init__(self) -> None:
        pass

    def refatorar_dataframe(self, df, colunas_agrupamento):
        df['dtfaturamento'] = pd.to_datetime(df['dtfaturamento'])
        cols_groupby = [pd.Grouper(key='dtfaturamento', freq='MS')] + colunas_agrupamento
        df_agrupado = df.groupby(cols_groupby)[['volume_t']].sum()
        df_full = df_agrupado.unstack(colunas_agrupamento).fillna(0).stack(colunas_agrupamento)
        df_full.reset_index(inplace=True)
        return df_full
    def carregar_dias_uteis(self, caminho_excel):
        """
        Carrega e prepara o dataframe de dias úteis a partir do Excel enviado.
        """
        try:
            df_dias = pd.read_excel(caminho_excel)
            
            colunas = df_dias.columns
            df_dias = df_dias.rename(columns={colunas[0]: 'ds', colunas[1]: 'dias_uteis'})
            df_dias['ds'] = pd.to_datetime(df_dias['ds']) + pd.offsets.MonthBegin(0)
            
            df_dias['dias_uteis'] = pd.to_numeric(df_dias['dias_uteis'], errors='coerce')
            
            df_dias = df_dias.drop_duplicates(subset=['ds'], keep='first')
            
            # print("Dias úteis carregados com sucesso.")
            return df_dias[['ds', 'dias_uteis']]
        except Exception as e:
            print(f"Erro ao carregar dias úteis: {e}")
            return pd.DataFrame(columns=['ds', 'dias_uteis'])
        
    def sasonalidade_ok(self, num_meses_ativo, meses_com_compra):
        if num_meses_ativo < 6: return False
        elif meses_com_compra / num_meses_ativo < 0.5: return False
        else: return True

    def criar_features_lags(self, df, janelas):
        cols_id = ['cliente_grupo', 'familia_repasse_1', 'cluster', 'grupo_tipologico']
        for mes in janelas:
            df[f'volume_lag_m{mes}'] = df.groupby(cols_id)['volume_t'].shift(mes)
        return df

    def criar_features_moveis(self, df, janelas):
        cols_id = ['cliente_grupo', 'familia_repasse_1', 'cluster', 'grupo_tipologico']
        df = df.sort_values(['dtfaturamento'])
        for mes in janelas:
            df[f'media_movel_m{mes}'] = df.groupby(cols_id)['volume_t'].transform(
                lambda x: x.shift(1).rolling(window=mes).mean()
            )
        return df.fillna(0)

    def gerar_probabilidade_compra(self, df_completo):
        print("   > Treinando Modelo de Probabilidade de Compra...")
        
        cols_agrup = ['cliente_grupo', 'familia_repasse_1', 'cluster', 'grupo_tipologico', 'uf']
        df = self.refatorar_dataframe(df_completo, cols_agrup)
        
        # Engenharia de Features (Lags/Médias)
        df = self.criar_features_lags(df, [1, 2, 3, 6])
        df = self.criar_features_moveis(df, [3, 6])
        df['mes'] = df.dtfaturamento.dt.month
        df['target'] = (df['volume_t'] > 0).astype(int)
        
        # Treino do Modelo
        drop_cols = ['dtfaturamento', 'volume_t', 'target', 'cliente_grupo', 'uf']
        df_model = pd.get_dummies(df, columns=['familia_repasse_1', 'grupo_tipologico', 'cluster'])
        df_train = df_model.dropna()
        
        X = df_train.drop(columns=drop_cols, errors='ignore')
        y = df_train['target']
        
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X, y)
        
        # Previsão para o próximo mês
        data_ref = df.dtfaturamento.max()
        df_last = df[df.dtfaturamento == data_ref].copy()
        
        # Dummies para previsão
        df_last_model = pd.get_dummies(df_last, columns=['familia_repasse_1', 'grupo_tipologico', 'cluster'])
        for col in X.columns:
            if col not in df_last_model.columns: df_last_model[col] = 0
        X_pred = df_last_model[X.columns]
        
        probs = model.predict_proba(X_pred)[:, 1]
        
        # Calcula a data da última compra REAL de cada cliente/família (ignorando zeros preenchidos)
        # Usamos o df original antes do refatorar para pegar as datas reais
        ultima_compra = df_completo[df_completo['volume_t'] > 0].groupby(['cliente_grupo', 'familia_repasse_1'])['dtfaturamento'].max().reset_index()
        ultima_compra.rename(columns={'dtfaturamento': 'data_ultima_venda'}, inplace=True)
        
        # Junta essa data na tabela de previsão
        df_result = df_last[['cliente_grupo', 'familia_repasse_1']].copy()
        df_result['prob_venda_ml'] = probs # Probabilidade pura do modelo
    
        df_result = df_result.merge(ultima_compra, on=['cliente_grupo', 'familia_repasse_1'], how='left')
        
        # Calcula dias de inatividade em relação à data de referência da previsão
        df_result['dias_inativo'] = (data_ref - df_result['data_ultima_venda']).dt.days
        df_result['dias_inativo'] = df_result['dias_inativo'].fillna(9999) # Nunca comprou = Infinito
        
        # Se inativo > 180 dias, Probabilidade = 0.0
        df_result['prob_venda'] = np.where(
            df_result['dias_inativo'] > 180, 
            0.0, 
            df_result['prob_venda_ml']
        )

        df_result['status_prob'] = np.where(
            df_result['dias_inativo'] > 180, 
            "INATIVO", 
            "ATIVO"
        )
        
        # Salvar parquet e excel dentro da pasta Pedido Sugerido/data/dados_processados/previsao
        df_result.to_parquet(Path(__file__).resolve().parent.parent.parent / 'data' / 'dados_processados' / 'previsao' / 'probabilidade_venda.parquet')
        df_result.to_excel(Path(__file__).resolve().parent.parent.parent / 'data' / 'dados_processados' / 'previsao' / 'probabilidade_venda.xlsx', index=False)
        

        return df_result#[['cliente_grupo', 'familia_repasse_1', 'prob_venda']]


    # =========================================================================
    # FORECASTING
    # =========================================================================

    def previsao_prophet(self, df, valor_alvo, coluna_alvo, colunas_filtro, PERIODOS_PREVISAO):

        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

        data_max = df.dtfaturamento.max()
        
        mask = (df[coluna_alvo] == valor_alvo)
        for col, val in colunas_filtro.items():
            mask = mask & (df[col] == val)
        dfp = df[mask].copy()

        if dfp.empty or dfp.volume_t.sum() == 0: return pd.DataFrame()

        data_min = dfp[(dfp.volume_t > 0)].dtfaturamento.min()
        if pd.isna(data_min): return pd.DataFrame()

        dfp = dfp[dfp.dtfaturamento >= data_max - relativedelta(months=18)].copy()

        if data_min < data_max - relativedelta(months=6):
            dfp = dfp[dfp.dtfaturamento >= data_min].copy()
        else:
            dfp = dfp[dfp.dtfaturamento >= data_max - relativedelta(months=6)].copy()

        df_feriados_csv = self.carregar_dias_uteis(Path(__file__).resolve().parent.parent.parent / 'data' / 'auxiliar' / 'dias_uteis_por_mes_recife.xlsx')
        dfp.rename(columns={'dtfaturamento': 'ds', 'volume_t': 'y'}, inplace=True)
        dfp = pd.merge(dfp, df_feriados_csv, on='ds', how='left')
        dfp['dias_uteis'] = dfp['dias_uteis'].fillna(21)
        # print(dfp.head(2))
        # 1. Defina o Piso (0) e o Teto (ex: dobro do máximo histórico) no DF de Treino
        # dfp['floor'] = 0
        # # valor teto maior que o máximo histórico 
        # if dfp['y'].max() == 0:
        #     cap_value = 100
        # else:
        #     cap_value = dfp['y'].max() * 2 
        # dfp['cap'] = cap_value

        
        model = Prophet(
            growth='linear',
            seasonality_mode='additive',
        )
        model.add_regressor('dias_uteis')
        
        try:
        
            model.fit(dfp[['ds', 'y', 'dias_uteis']])#, 'floor', 'cap']])
            future = model.make_future_dataframe(periods=PERIODOS_PREVISAO, freq='MS')
            future = pd.merge(future, df_feriados_csv, on='ds', how='left')
            future['dias_uteis'] = future['dias_uteis'].fillna(21)
            # future['floor'] = 0
            # future['cap'] = cap_value 
            forecast = model.predict(future)
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            # Compara o Histórico Real (dfp) com o Ajuste do Modelo (forecast) nas mesmas datas
            historico_check = dfp.merge(forecast[['ds', 'yhat', 'dias_uteis']], on='ds', how='inner')
            
            # Calculando W-MAPE (Weighted MAPE): Soma do Erro Absoluto / Soma do Volume Total
            soma_erro = np.sum(np.abs(historico_check['y'] - historico_check['yhat']))
            soma_real = np.sum(historico_check['y'])
            
            if soma_real > 0:
                wmape = soma_erro / soma_real
            else:
                wmape = 0.0
            
            # Adiciona a métrica como uma coluna constante (repete em todas as linhas)
            forecast['erro_modelo'] = wmape
            
        except Exception as e:
            print("Erro no modelo Prophet para:", coluna_alvo, valor_alvo)
            print(e)
            return pd.DataFrame()

        forecast.rename(columns={'ds': 'dtfaturamento', 'yhat': 'volume_t'}, inplace=True)
        forecast = forecast[forecast.dtfaturamento > data_max].copy()
        
        forecast[coluna_alvo] = valor_alvo
        for col, val in colunas_filtro.items():
            forecast[col] = val
            
        # return forecast[['dtfaturamento', 'volume_t', coluna_alvo] + list(colunas_filtro.keys())]
        cols_retorno = ['dtfaturamento', 'volume_t', coluna_alvo, 'erro_modelo'] + list(colunas_filtro.keys())
        # print(forecast[cols_retorno].head())
        return forecast[cols_retorno]

    # =========================================================================
    # PREVISOES
    # =========================================================================

    def gerar_previsoes_base(self, df, colunas_agrupamento, coluna_alvo, PERIODOS_PREVISAO):
        """Gera previsão Macro (Gerente). Usa Prophet ou Média."""
        print(f"   > Refatorando dados para: {colunas_agrupamento}...")
        df_ref = self.refatorar_dataframe(df, colunas_agrupamento)
        chaves_unicas = df_ref[colunas_agrupamento].drop_duplicates()
        df_final = pd.DataFrame()
        
        for i, (_, row_key) in enumerate(chaves_unicas.iterrows()):
            if i % 100 == 0: print(f"     Processando item {i}/{len(chaves_unicas)}...")
            
            valor_alvo = row_key[coluna_alvo]
            outras_cols = {c: row_key[c] for c in colunas_agrupamento if c != coluna_alvo}
            
            mask = (df_ref[coluna_alvo] == valor_alvo)
            for c, v in outras_cols.items(): mask = mask & (df_ref[c] == v)
            df_serie = df_ref[mask]
            if df_serie.volume_t.sum() == 0: continue

            data_min = df_serie[df_serie.volume_t > 0].dtfaturamento.min()
            if pd.isna(data_min): continue
            meses_ativo = (df_ref.dtfaturamento.max() - data_min) / np.timedelta64(30, 'D')
            meses_com_compra = len(df_serie[df_serie.volume_t > 0])
            
            forecast = pd.DataFrame()
            if self.sasonalidade_ok(meses_ativo, meses_com_compra):
                forecast = self.previsao_prophet(df_ref, valor_alvo, coluna_alvo, outras_cols, PERIODOS_PREVISAO)
            
            if forecast.empty:
                media = df_serie[df_serie.volume_t > 0].volume_t.mean()
                if pd.isna(media): media = 0
                datas = pd.date_range(
                    start=df_ref.dtfaturamento.max() + relativedelta(months=1), 
                    periods=PERIODOS_PREVISAO, freq='MS'
                    )
                forecast = pd.DataFrame({'dtfaturamento': datas, 'volume_t': media})
                forecast[coluna_alvo] = valor_alvo
                for c, v in outras_cols.items(): forecast[c] = v
                forecast['erro_modelo'] = np.nan
            
            if not forecast.empty: df_final = pd.concat([df_final, forecast])

        return df_final

    def gerar_previsoes_detalhadas(self, df, colunas_agrupamento, coluna_alvo, PERIODOS_PREVISAO):
        """Gera previsão Micro (Cliente) com a regra de FORÇAR MÉDIA."""
        df_ref = self.refatorar_dataframe(df, colunas_agrupamento)
        chaves_unicas = df_ref[colunas_agrupamento].drop_duplicates()
        df_final = pd.DataFrame()
        
        # Pré-cálculo de média 6m para a regra de FORÇAR MÉDIA
        data_corte = df.dtfaturamento.max() - relativedelta(months=6)
        df_recente = df_ref[df_ref.dtfaturamento >= data_corte]
        medias_6m = df_recente.groupby(colunas_agrupamento)['volume_t'].mean()
#         gerente_contas familia_repasse_1                                    
# CB2            Pintado               1.58     1.58     1.58     1.58
# CC1            Pintado               7.91     7.91     7.91     7.91

        # Só fazer para os Gerentes CB2 família Pintado e CC1 família Pintado
        # chaves_unicas = chaves_unicas[
        #     (chaves_unicas.gerente_contas.isin(['CB2', 'CC1'])) 
        #     # & (chaves_unicas.familia_repasse_1 == 'Pintado')
        # ]
        
        for i, (_, row_key) in enumerate(chaves_unicas.iterrows()):
            if i % 100 == 0: print(f"     Processando item {i}/{len(chaves_unicas)}...")
            
            valor_alvo = row_key[coluna_alvo]
            outras_cols = {c: row_key[c] for c in colunas_agrupamento if c != coluna_alvo}
            mask = (df_ref[coluna_alvo] == valor_alvo)
            for c, v in outras_cols.items(): mask = mask & (df_ref[c] == v)
            df_serie = df_ref[mask]
            if df_serie.volume_t.sum() == 0: continue

            try:
                forecast = self.previsao_prophet(df_ref, valor_alvo, coluna_alvo, outras_cols, PERIODOS_PREVISAO)
            except: forecast = pd.DataFrame()

            if forecast.empty:
                datas = pd.date_range(start=df_ref.dtfaturamento.max() + relativedelta(months=1), periods=PERIODOS_PREVISAO, freq='MS')
                forecast = pd.DataFrame({'dtfaturamento': datas, 'volume_t': 0})
                forecast[coluna_alvo] = valor_alvo
                for c, v in outras_cols.items(): forecast[c] = v
                forecast['erro_modelo'] = np.nan

            forecast.rename(columns={'volume_t': 'volume_prophet'}, inplace=True)
            forecast['volume_prophet'] = forecast['volume_prophet'].clip(lower=0)

            try:
                chave = tuple(row_key[col] for col in medias_6m.index.names)
                media_val = medias_6m.loc[chave]
            except KeyError: media_val = 0
            
       
            # Regra: Se Prophet Zero e Média Positiva, usa Média     
            forecast['volume_forcado'] = np.where(
                (forecast['volume_prophet'] < 0.01) & (media_val > 0),
                media_val,
                forecast['volume_prophet']
            )

            # Cria a coluna se usou ou não a média
            forecast['usou_media'] = (forecast['volume_prophet'] < 0.01) & (media_val > 0)
            forecast['media_6m'] = media_val
            # print(f"     Cliente: {valor_alvo} | {forecast['volume_prophet'].iloc[0]:.2f}, {media_val:.2f}, {forecast['volume_forcado'].iloc[0]:.2f}")

            
            forecast['gordura'] = forecast['volume_forcado'] - forecast['volume_prophet']
            df_final = pd.concat([df_final, forecast])
        return df_final

    # =========================================================================
    # EXECUÇÃO
    # =========================================================================

    def executar(self, mes_ref_str, N_CLUSTERS, PERIODOS_PREVISAO):
        
        # --- SETUP ---
        # mes_ref = datetime.strptime(mes_ref_str, "%Y-%m")
        try: SCRIPT_DIR = Path(__file__).resolve().parent 
        except NameError: SCRIPT_DIR = Path.cwd() 
        PROJECT_ROOT = SCRIPT_DIR.parent.parent 
        DATA_DIR = PROJECT_ROOT / 'data'
        
        print("1. Carregando Dados...")
        df = pd.read_parquet(DATA_DIR / 'dados_processados/base_dados_pedido_sugerido.parquet')
        resultados = pd.read_parquet(DATA_DIR / f'dados_processados/pedido_sugerido_n{N_CLUSTERS}.parquet')
        # Remover familia_repasse_1 == 'familia_repasse_1' do df e do resultados
        df = df[df['familia_repasse_1'] != 'familia_repasse_1']
        resultados = resultados[resultados['familia_repasse_1'] != 'familia_repasse_1']

        
        res_reset = resultados.reset_index()
        res_reset.cluster = res_reset.cluster.astype(str)
        d_clust = dict(zip(res_reset['cliente_grupo'], res_reset['cluster']))
        df['cluster'] = df['cliente_grupo'].map(d_clust)
        df.dropna(subset=['cluster'], inplace=True)
        df.cluster = pd.to_numeric(df.cluster).astype(int)
        df['dtfaturamento'] = pd.to_datetime(df['dtfaturamento'])

        # --- PREVISÃO MACRO E MICRO ---
        print("\n2. Calculando Previsões...")
        cols_macro = ['gerente_contas', 'familia_repasse_1']
        df_macro = self.gerar_previsoes_base(df, cols_macro, 'gerente_contas', PERIODOS_PREVISAO)
        df_macro.rename(columns={'volume_t': 'volume_meta'}, inplace=True)

        cols_micro = ['cliente_grupo', 'gerente_contas', 'familia_repasse_1', 'cluster', 'uf']
        df_micro = self.gerar_previsoes_detalhadas(df, cols_micro, 'cliente_grupo', PERIODOS_PREVISAO)

        # --- PROBABILIDADE ---
        print("\n3. Calculando Probabilidades...")
        df_probs = self.gerar_probabilidade_compra(df)
        df_micro = df_micro.merge(df_probs, on=['cliente_grupo', 'familia_repasse_1'], how='left')
        df_micro['prob_venda'] = df_micro['prob_venda'].fillna(-1) # Nulos vão para o fim da fila

        # --- RECONCILIAÇÃO DE CAIXA ---
        print("\n4. Aplicando Corte e Acréscimo...")
        
        # Junta Meta do Gerente
        df_final = df_micro.merge(
            df_macro[['gerente_contas', 'familia_repasse_1', 'dtfaturamento', 'volume_meta']],
            on=['gerente_contas', 'familia_repasse_1', 'dtfaturamento'],
            how='left'
        )
        df_final['volume_meta'].fillna(0, inplace=True)

        # Agrupamos por Gerente/Família/Data e ordenamos por Probabilidade (Maior -> Menor)
        df_final = df_final.sort_values(
            by=['gerente_contas', 'familia_repasse_1', 'dtfaturamento', 'prob_venda'],
            ascending=[True, True, True, False]
        )

        # Acumulado
        cols_grupo = ['gerente_contas', 'familia_repasse_1', 'dtfaturamento']
        df_final['volume_acumulado'] = df_final.groupby(cols_grupo)['volume_forcado'].cumsum()
        df_final['volume_acumulado_anterior'] = df_final.groupby(cols_grupo)['volume_acumulado'].shift(1).fillna(0)

        # Se sobrar volume na meta, o cliente pega. Se acabar, ele fica com 0.
        # Se a meta for MAIOR que a soma (Falta), isso aqui não corta ninguém (todos pegam tudo).
        df_final['volume_pos_corte'] = np.minimum(
            df_final['volume_forcado'], 
            df_final['volume_meta'] - df_final['volume_acumulado_anterior']
        ).clip(lower=0)


        # Garantir que o volume pela média entre no volume_pos_corte para aqueles com volume_forcado > volume_prophet
        df_final.loc[df_final['volume_forcado'] > df_final['volume_prophet'], 'volume_pos_corte'] = df_final['volume_forcado']
        # print(df_final[['cliente_grupo', 'volume_prophet', 'volume_forcado', 'volume_pos_corte', 'volume_meta', 'foi_cortado']].head(50).to_string())
        
        # Recalcula a soma 
        soma_pos_corte = df_final.groupby(cols_grupo)['volume_pos_corte'].transform('sum')
        
        # Fator = Meta / Soma Pós Corte. 
        # Se Soma > Meta (Houve Excesso), a cascata já resolveu, a soma_pos_corte == Meta, fator = 1.
        # Se Soma < Meta (Houve Falta), a cascata não fez nada, soma_pos_corte < Meta, fator > 1.
        # df_final['fator_multiplicativo'] = np.where(
        #     (soma_pos_corte > 0) & (soma_pos_corte < df_final['volume_meta'] - 0.01),
        #     (df_final['volume_meta'] / soma_pos_corte).round(2),
        #     1.0
        # )
        
        
        # APLICAÇÃO FINAL
        # df_final['volume_final'] = df_final['volume_pos_corte'] * df_final['fator_multiplicativo']
        df_final['volume_final'] = df_final['volume_pos_corte']* np.where(
            soma_pos_corte > 0,
            (df_final['volume_meta'] / soma_pos_corte).round(2),
            1.0
        )
        # --- MÉTRICAS FINAIS ---
        df_final['volume_cortado'] = df_final['volume_forcado'] - df_final['volume_pos_corte']
        df_final['volume_adicionado'] = df_final['volume_final'] - df_final['volume_pos_corte']
        
        # fator corte, indica o quanto do volume final foi cortado, 
        # se o volume for cortado totalmente, o fator é 0
        # se o volume não for cortado, o fator é 1
        # se o volume for aumentado, o fator é > 1 (ex: 1.2 = aumento de 20%)
        df_final['fator_corte'] = np.where(
            df_final['volume_forcado'] > 0,
            (df_final['volume_final'] / df_final['volume_forcado']).round(2),
            0.0
        )
  

        print("\n6. Salvando Arquivos...")
        # suffix = mes_ref.strftime("%Y-%m")
        df_final['mes_ref'] = df_final['dtfaturamento'].dt.strftime('%Y-%m')
        df_macro['mes_ref'] = df_macro['dtfaturamento'].dt.strftime('%Y-%m')

        # CLIENTE
        if 'prob_venda' in df_final.columns:
            df_final['prob_venda'] = df_final['prob_venda'].fillna(-1)

        export_cliente = df_final.pivot_table(
            index=['cliente_grupo', 'gerente_contas', 'familia_repasse_1', 'cluster', 'uf',
                    'dias_inativo', 'status_prob', 'prob_venda', 'erro_modelo', 
                    'media_6m', 'usou_media'],
            columns='mes_ref',
            values=['volume_final', 
                    # 'foi_cortado', 
                    # 'fator_multiplicativo',
                    'fator_corte', 
                    # 'foi_adicionado', 
                    # 'pct_adicionado'
                    ],
            aggfunc={
                'volume_final':'sum', 
                # 'fator_multiplicativo':'first',
                'fator_corte':'first',
                # 'foi_cortado':'first',
                # 'pct_corte':'first',
                # 'foi_adicionado':'first',
                # 'pct_adicionado':'first'
            },
            fill_value=0
        )
        
        novas_colunas = []
        for col in export_cliente.columns:
            var, mes = col
            if var == 'volume_final': alias = 'volume_t'
            else: alias = var
            novas_colunas.append(f"{alias}_{mes}")
        export_cliente.columns = novas_colunas
        export_cliente.reset_index(inplace=True)
        # Dia atual em que o código foi rodado 
        day = datetime.now().strftime("%Y-%m-%d")
        
        run = 1
        # Define os nomes propostos
        nome_cliente = f"previsao_volume_{mes_ref_str}_CLIENTE_{day}_run{run}.xlsx"
        nome_gerente = f"previsao_volume_{mes_ref_str}_GERENTE_{day}_run{run}.xlsx"
        
        path_cliente = DATA_DIR / 'dados_processados' / 'previsao' / nome_cliente
        path_gerente = DATA_DIR / 'dados_processados' / 'previsao' / nome_gerente
        
        
        # while True:
        #     # Define os nomes propostos
        #     nome_cliente = f"previsao_volume_{mes_ref_str}_CLIENTE_{day}_run{run}.xlsx"
        #     nome_gerente = f"previsao_volume_{mes_ref_str}_GERENTE_{day}_run{run}.xlsx"
            
        #     path_cliente = DATA_DIR / 'dados_processados' / 'previsao' / nome_cliente
        #     path_gerente = DATA_DIR / 'dados_processados' / 'previsao' / nome_gerente
            
        #     # A Lógica: Se ALGUM dos dois já existir, incrementa o run e tenta de novo
        #     if os.path.exists(path_cliente) or os.path.exists(path_gerente):
        #         run += 1
        #     else:
        #         break

        export_cliente.to_excel(path_cliente, index=False)
        print(f"Arquivo Cliente salvo: {path_cliente}")

        # GERENTE
        export_gerente = df_macro.pivot_table(
            index=['gerente_contas', 'familia_repasse_1'], columns='mes_ref',
            values='volume_meta', aggfunc='sum', fill_value=0
        ).reset_index()
        export_gerente.to_excel(path_gerente, index=False)
        print(f"Arquivo Gerente salvo: {path_gerente}")
        


        return export_gerente, export_cliente

if __name__ == "__main__":
    previsor = previsao_volume()
    previsor.executar(
        mes_ref_str="2025-11",
        N_CLUSTERS=7,
        PERIODOS_PREVISAO=3
    )

