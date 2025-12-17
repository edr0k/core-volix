import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from dateutil.relativedelta import relativedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class validacao:
    def __init__(self, path_cliente, path_gerente, path_historico):
        """
        Inicializa a classe de validação com os caminhos dos arquivos.
        """
        self.path_cliente = path_cliente
        self.path_gerente = path_gerente
        self.path_historico = path_historico
        
        self.df_cliente = None
        self.df_gerente = None
        self.df_historico = None
        self.df_final_long = None

    def carregar_dados(self):
        """Carrega os arquivos Excel e Parquet."""
        print("--- Carregando dados ---")
        
        if str(self.path_historico).endswith('.parquet'):
            self.df_historico = pd.read_parquet(self.path_historico)
        else:
            self.df_historico = pd.read_excel(self.path_historico)
            
        self.df_cliente = pd.read_excel(self.path_cliente)
        self.df_gerente = pd.read_excel(self.path_gerente)
        
        if 'dtfaturamento' in self.df_historico.columns:
            self.df_historico['dtfaturamento'] = pd.to_datetime(self.df_historico['dtfaturamento'])

        print(f"Dados carregados. Histórico: {self.df_historico.shape}, Cliente: {self.df_cliente.shape}, Gerente: {self.df_gerente.shape}")

    def transformar_previsoes_em_linhas(self, df_wide, data_inicio_previsao):
        """
        Transforma dataframe largo para longo, protegendo colunas de texto para não serem somadas.
        """
        lista_ids_fixos = ['cliente_grupo', 'gerente_contas', 'familia_repasse_1', 'uf', 'regiao_2', 'cluster', 'setor']
        
        cols_id = [c for c in df_wide.columns if c in lista_ids_fixos or df_wide[c].dtype == 'object']
        
        ignore_cols = cols_id + ['usou_media', 'erro_modelo']
        cols_valores = [c for c in df_wide.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_wide[c])]
        
        print(f"Colunas identificadas como volume (serão somadas): {cols_valores}")
        
        df_long = pd.melt(
            df_wide,
            id_vars=cols_id,
            value_vars=cols_valores,
            var_name='mes_ref',
            value_name='volume'
        )
        
        lista_cols = sorted(cols_valores)
        mapa_datas = {}
        for i, col in enumerate(lista_cols):
            # Tenta extrair data se o nome da coluna contiver '202' (ex: volume_2025-01)
            # Se não, usa a lógica sequencial a partir da data de início
            try:
                # Tenta achar algo com cara de data no nome da coluna
                import re
                match = re.search(r'(\d{4}-\d{2})', str(col))
                if match:
                    data_str = match.group(1) + '-01'
                    mapa_datas[col] = pd.to_datetime(data_str)
                else:
                    raise ValueError("Sem data no nome")
            except:
                # Fallback: Sequencial
                mapa_datas[col] = data_inicio_previsao + relativedelta(months=i)

        df_long['ds'] = df_long['mes_ref'].map(mapa_datas)
        df_long = df_long.drop(columns=['mes_ref'])
        
        df_long['volume'] = pd.to_numeric(df_long['volume'], errors='coerce').fillna(0)
        
        return df_long

    def gerar_analise_residuos(self, data_analise_str='2024-11-01', col_previsao_alvo='volume_t_2025-11'):
        """
        Cruza o realizado de um mês específico (no passado) com a previsão gerada para ele (teste retroativo)
        ou compara com um benchmark.
        """
        print(f"--- Gerando Análise de Resíduos para {data_analise_str} ---")
        
        data_analise = pd.to_datetime(data_analise_str)
        mes_fim = data_analise + relativedelta(months=1)
        
        mask_hist = (self.df_historico['dtfaturamento'] >= data_analise) & (self.df_historico['dtfaturamento'] < mes_fim)
        
        df_real = self.df_historico[mask_hist].groupby(['cliente_grupo', 'familia_repasse_1']).agg({'volume_t':'sum'}).reset_index()
        df_real.rename(columns={'volume_t': 'volume_real'}, inplace=True)
        
        if col_previsao_alvo not in self.df_cliente.columns:
            print(f"AVISO: Coluna {col_previsao_alvo} não encontrada no arquivo de cliente. Pulando análise de resíduos.")
            return

        df_prev = self.df_cliente.groupby(['cliente_grupo', 'familia_repasse_1']).agg({col_previsao_alvo:'sum'}).reset_index()
        df_prev.rename(columns={col_previsao_alvo: 'volume_previsto'}, inplace=True)
        
        df_analise = pd.merge(df_real, df_prev, on=['cliente_grupo', 'familia_repasse_1'], how='outer').fillna(0)
        
        df_analise['residuo'] = df_analise['volume_previsto'] - df_analise['volume_real']
        df_analise['pct_erro'] = np.where(df_analise['volume_real'] != 0, 
                                          df_analise['residuo'] / df_analise['volume_real'], 0)
        
        output_path = DATA_DIR / f'analise_residuos_{data_analise.strftime("%Y-%m")}.xlsx'
        df_analise.to_excel(output_path, index=False)
        print(f"Análise de resíduos salva em: {output_path}")
        
        return df_analise

    def gerar_graficos_gerente(self, data_inicio_previsao, nome_arquivo_pdf="Relatorio_Gerentes.pdf"):
        """
        Gera o PDF com gráficos de linha por gerente (Realizado + Previsto).
        """
        print("--- Preparando dados para gráficos ---")
        
        # 1. Transforma previsão cliente para longo
        df_long = self.transformar_previsoes_em_linhas(self.df_gerente, data_inicio_previsao)
        
        if 'gerente_contas' not in df_long.columns and 'cliente_grupo' in df_long.columns:
            # Tenta recuperar do histórico (mapa de clientes)
            mapa_gerentes = self.df_historico[['cliente_grupo', 'gerente_contas']].drop_duplicates()
            df_long = df_long.merge(mapa_gerentes, on='cliente_grupo', how='left')

        self.df_historico = self.df_historico[self.df_historico['familia_repasse_1'] != 'familia_repasse_1']  # Remove registros inválidos
        print(self.df_historico.dtfaturamento.max())

        # cols_hist = ['dtfaturamento', 'gerente_contas', 'familia_repasse_1', 'volume_t']
        df_hist_plot = self.df_historico.copy()
        df_hist_plot['dtfaturamento'] = pd.to_datetime(df_hist_plot['dtfaturamento']) + pd.offsets.MonthBegin(0)
        df_hist_plot['mes'] = pd.to_datetime(df_hist_plot.dtfaturamento).dt.month
        df_hist_plot['ano'] = pd.to_datetime(df_hist_plot.dtfaturamento).dt.year
        df_hist_plot = df_hist_plot.groupby(['mes', 'ano', 'gerente_contas', 'familia_repasse_1']).agg({'volume_t':'sum', 'dtfaturamento':'first'}).reset_index() 
        df_hist_plot['Tipo_Dado'] = 'Realizado'
        df_hist_plot.rename(columns={'dtfaturamento': 'ds', 'volume_t': 'volume'}, inplace=True)
        
        df_prev_plot = df_long.copy()
        df_prev_plot = df_prev_plot[df_prev_plot.familia_repasse_1!='familia_repasse_1']  # Remove previsões sem família
        df_prev_plot['Tipo_Dado'] = 'Previsto'
        # Garante as mesmas colunas
        cols_plot = ['ds', 'gerente_contas', 'familia_repasse_1', 'volume', 'Tipo_Dado']
            # Verifica se as colunas existem (segurança)
        for c in cols_plot:
            if c not in df_hist_plot.columns or c not in df_prev_plot.columns:
                print(f"ERRO: Coluna '{c}' não encontrada. Verifique sua query SQL.")
                return
        # Agrupa por Gerente e Família e Data (para somar todos os clientes de um gerente)
        # df_hist_agg = df_hist_plot.groupby(['ds', 'gerente_contas', 'familia_repasse_1', 'Tipo_Dado'])['volume'].sum().reset_index()
        # df_prev_agg = df_prev_plot.groupby(['ds', 'gerente_contas', 'familia_repasse_1', 'Tipo_Dado'])['volume'].sum().reset_index()
        
        df_full = pd.concat([df_hist_plot, df_prev_plot], ignore_index=True).sort_values('ds')
        
        # 4. Plotagem
        print(f"--- Gerando PDF: {nome_arquivo_pdf} ---")
        pdf = PdfPages(nome_arquivo_pdf)
        lista_gerentes = df_full['gerente_contas'].dropna().unique()
        
        for gerente in lista_gerentes:
            dados = df_full[df_full['gerente_contas'] == gerente]
            if dados.empty: continue
            
            plt.figure(figsize=(14, 7))
            sns.lineplot(
                data=dados, x='ds', y=dados.volume/dados.volume.max(), hue='familia_repasse_1', style='Tipo_Dado',
                markers=True, dashes={'Realizado': (None, None), 'Previsto': (2, 2)}
            )
            
            plt.title(f'Volume: {gerente}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.axvline(x=df_hist_plot['ds'].max(), color='red', linestyle='--', alpha=0.5)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
        pdf.close()
        print("PDF gerado com sucesso.")

    def verificar_metricas_medias(self):
        """
        Verifica quantos clientes tiveram a previsão feita pela média simples (fallback).
        """
        if 'usou_media' not in self.df_cliente.columns:
            print("Coluna 'usou_media' não encontrada para cálculo de métricas.")
            return

        total = len(self.df_cliente)
        qtd_media = self.df_cliente['usou_media'].sum()
        pct = (qtd_media / total) * 100
        
        print(f"\n--- Métricas de Modelo ---")
        print(f"Total de previsões: {total}")
        print(f"Previsões via Média (Fallback): {qtd_media} ({pct:.2f}%)")

# --- BLOCO DE EXECUÇÃO (Main) ---
if __name__ == "__main__":
    # --- SETUP ---
    # mes_ref = datetime.strptime(mes_ref_str, "%Y-%m")
    try: SCRIPT_DIR = Path(__file__).resolve().parent 
    except NameError: SCRIPT_DIR = Path.cwd() 
    PROJECT_ROOT = SCRIPT_DIR.parent.parent 
    DATA_DIR = PROJECT_ROOT / 'data'
    mes_ref = '2025-11' # último mes de fechamento
    path_cli = DATA_DIR / f'dados_processados/previsao/previsao_volume_{mes_ref}_CLIENTE_2025-12-15_run1.xlsx'
    path_ger = DATA_DIR / f'dados_processados/previsao/previsao_volume_{mes_ref}_GERENTE_2025-12-15_run1.xlsx'
    path_hist = DATA_DIR / 'dados_processados/pedido_sugerido_n7.parquet' # ou .xlsx
    
    validador = validacao(path_cli, path_ger, path_hist)
    
    validador.carregar_dados()
    
    validador.verificar_metricas_medias()
    
    # Ajuste 'col_previsao_alvo' para o nome da coluna que representa o mês que você quer validar
    validador.gerar_analise_residuos(data_analise_str=mes_ref, col_previsao_alvo='volume_t_2025-11')
    
    # Define quando começa a previsão (ex: Próximo mês)
    inicio_previsao = pd.to_datetime(mes_ref)+relativedelta(months=1)
    validador.gerar_graficos_gerente(data_inicio_previsao=inicio_previsao, nome_arquivo_pdf=DATA_DIR / f'Relatorio_Validacao_Gerentes_{mes_ref}.pdf')