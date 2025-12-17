import json
from matplotlib.dates import relativedelta
import pandas as pd
from pathlib import Path

from scripts.ETL import ETL
from scripts.perfil import perfil
from scripts.previsao_volume import previsao_volume
from scripts.elasticidade import elasticidade
from scripts.validacao import validacao


mes_ref = '2025-11'
inicio_previsao = pd.to_datetime(mes_ref)+relativedelta(months=1)
N_CLUSTERS = 7
PERIODOS_PREVISAO = 3

try: SCRIPT_DIR = Path(__file__).resolve().parent 
except NameError: SCRIPT_DIR = Path.cwd() 
PROJECT_ROOT = SCRIPT_DIR.parent.parent 
DATA_DIR = PROJECT_ROOT / 'data'

path_cli = DATA_DIR / f'dados_processados/previsao/previsao_volume_{mes_ref}_CLIENTE_2025-12-15_run1.xlsx'
path_ger = DATA_DIR / f'dados_processados/previsao/previsao_volume_{mes_ref}_GERENTE_2025-12-15_run1.xlsx'
path_hist = DATA_DIR / 'dados_processados/pedido_sugerido_n7.parquet' # ou .xlsx

etl = ETL()
perfil = perfil()
previsao_volume = previsao_volume()
# elasticidade = elasticidade()
# validacao = validacao(path_cli, path_ger, path_hist)

print('------------------------- ETL ------------------------------------')
etl.executar_clientes()
etl.executar_materiais()
etl.executar_faturamento(mes_ref)

print('------------------------- perfis ------------------------------------')
perfil.executar()
print('------------------------- previsao_volume ------------------------------------')
previsao_volume.executar(mes_ref,N_CLUSTERS,PERIODOS_PREVISAO)
# print('------------------------- elasticidade ------------------------------------')
# elasticidade.executar()
print('------------------------- validacao ------------------------------------')
# validacao.carregar_dados()
# validacao.gerar_graficos_gerente(
#     data_inicio_previsao=inicio_previsao,
#     nome_arquivo_pdf=DATA_DIR / f'Relatorio_Validacao_Gerentes_{mes_ref}.pdf')