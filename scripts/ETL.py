import pandas as pd
from unidecode import unidecode
import re
import datetime
from glob import glob
import json
from pathlib import Path

try: SCRIPT_DIR = Path(__file__).resolve().parent 
except NameError: SCRIPT_DIR = Path.cwd() 
PROJECT_ROOT = SCRIPT_DIR.parent.parent 
DATA_DIR = PROJECT_ROOT / 'data'

class ETL:
    def __init__(self) -> None:
        pass
        
    def data_arquivo(self,data_str):
        mes = datas[data_str.split('_')[-1].split('.')[0][:3]]
        ano = int(data_str.split('_')[-1].split('.')[0][3:])
        return  datetime.datetime(ano,mes,1)
    
    def ajusta_nome_colunas (self,coluna):
        coluna = coluna.lower().strip()
        dAjuste = {"%":"pct",
                " ":"_"}
        
        for de, para in dAjuste.items():
            coluna = coluna.replace(de, para)
            coluna=re.sub(r'[^\w ]',"",coluna)
            coluna = unidecode(coluna)

        return coluna

    def extrai_data_arquivo(self,nome_arquivo):
        meses = {
            "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
            "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
        }
        mes_ano = nome_arquivo.split('_')[-1].split('.')[0]
        return datetime.datetime(int(mes_ano[3:]),meses[mes_ano[:3]],1)

    def executar_clientes(self):
        
        arquivos = glob(str(DATA_DIR / 'dados_cliente/cliente/Base Clientes_Pedido Sugerido*.xlsx'))
        # print(arquivos)

        ultimo_mes = max([self.extrai_data_arquivo(x) for x in arquivos])
        ultimo_arquivo = [x for x in arquivos if self.extrai_data_arquivo(x)== ultimo_mes][0]


        df_clientes =  pd.read_excel(ultimo_arquivo)
        df_clientes.columns = [self.ajusta_nome_colunas(x) for x in df_clientes.columns]
        df_clientes.cliente_cod = df_clientes.cliente_cod.astype(int)
        df_clientes.drop_duplicates(subset=['cliente_cod'],keep='last', inplace=True)

        df_clientes = df_clientes[['cliente_cod',
                    'cliente',
                    'cliente_grupo',
                    'uf',
                    'gerente_comercial',
                    'gerente_contas',
                    'tipologia']].copy()

        df_clientes.to_parquet(str(DATA_DIR / 'dados_processados/base_clientes.parquet'))
    def executar_materiais(self):
        arquivos = glob(str(DATA_DIR / 'dados_cliente/materiais/Base Materiais_Pedido Sugerido*.xlsx'))
        
        ultimo_mes = max([self.extrai_data_arquivo(x) for x in arquivos])
        ultimo_arquivo = [x for x in arquivos if self.extrai_data_arquivo(x)== ultimo_mes][0]


        df_materiais =  pd.read_excel(ultimo_arquivo)
        df_materiais.columns = [self.ajusta_nome_colunas(x) for x in df_materiais.columns]
        #df_materiais.dropna(subset=['grupo_materiais'],inplace=True)
        #df_materiais = df_materiais[df_materiais.tipo!='Transição'].copy()
        df_materiais = df_materiais[['material_sku', 'material', 'familia', 'grupo_analise', 'cor',
            'espessura', 'dimensao','produto', 'familia_repasse_1', 'familia_repasse_2']].copy()

        df_materiais['material_sku'] = df_materiais['material_sku'].astype(int)

        for col in df_materiais.columns:
            if col != 'material_sku':
                df_materiais[col] = df_materiais[col].astype(str).fillna('-')

        df_materiais.to_parquet(str(DATA_DIR / 'dados_processados/base_materiais.parquet'))
    
    def executar_faturamento(self,mes_ref):
        # with open('parametros.json') as file:
        #     API_CLIENTE = json.load(file)

        arquivo_base = str(DATA_DIR / "dados_cliente/vendas/Base de Dados_Pedido Sugerido_2014 a 10_2023.xlsx")
        arquivos = glob(str(DATA_DIR / "dados_cliente/vendas/Base de Dados_Pedido*.xlsx"))
        df_grupo_tipologico = pd.read_parquet(str(DATA_DIR / 'dados_cliente/de_para/grupo_tipologico.parquet'))

        arquivo_clientes = str(DATA_DIR / 'dados_processados/base_clientes.parquet')
        arquivo_materiais = str(DATA_DIR / 'dados_processados/base_materiais.parquet')
        de_para_regioes = str(DATA_DIR / 'dados_cliente/de_para/regioes.xlsx')
        local_destino = str(DATA_DIR / 'dados_processados/base_dados_pedido_sugerido.parquet')

        cols_concat = ['anomes_fatura',
        'dtfaturamento',
        'nota_fiscal',
        'ordem_venda',
        'tp_doc_faturamento',
        'transporte',
        'cliente_cod',
        #'cliente',
        'material_sku',
        # 'material',
        'volume_t',
        'preco_medio_liq_rt',
        'receita_liquida']

        datas ={'Jan':1,
                'Fev':2,
                'Mar':3,
                'Abr':4,
                'Mai':5,
                'Jun':6,
                'Jul':7,
                'Ago':8,
                'Set':9,
                'Out':10,
                'Nov':11,
                'Dez':12
                }


        df_materiais = pd.read_parquet(arquivo_materiais)
        df_clientes =  pd.read_parquet(arquivo_clientes)

        



        df_base = pd.DataFrame()

        for arquivo in arquivos:

            df_base_i = pd.read_excel(
                arquivo,
                #sheet_name='Base',
                dtype = {'DtFaturamento':str}
                )
            df_base_i.columns = [self.ajusta_nome_colunas(x) for x in df_base_i.columns]

            df_base_i.rename(columns= {'prazo':'transporte'}, inplace=True)
            df_base_i = df_base_i[cols_concat].copy()
            df_base = pd.concat([df_base,df_base_i])

        datas1 = pd.to_datetime(df_base.dtfaturamento, format='%Y-%m-%d 00:00:00', errors='coerce')
        datas2 = pd.to_datetime(df_base.dtfaturamento, format='%d/%m/%Y', errors='coerce')

        df_base['dtfaturamento'] =  datas1.fillna(datas2)
        #df_clientes =  pd.read_excel(arquivo_base,sheet_name='Base_Clientes')

        

        df_base = df_base.merge(df_clientes, on =['cliente_cod'], how='left')
        df_base.shape

        regioes = pd.read_excel(de_para_regioes)
        regioes.columns = [self.ajusta_nome_colunas(x) for x in regioes.columns]
        regioes.uf.nunique(), len(regioes)
        df_base.shape
        df = df_base.merge(regioes[['uf', 'regiao_2']], on = 'uf', how='left')
        df.shape
        df.cliente_cod = df.cliente_cod.astype(int)


        

        df = df.merge(df_materiais, on = 'material_sku', how='left')

        df.shape


        df_grupo_tipologico.cliente_cod.nunique(),len(df_grupo_tipologico)

        df = df.merge(df_grupo_tipologico[['cliente_cod','grupo_tipologico']], on = 'cliente_cod', how='left')
        df.shape
        #df['dtfaturamento'] = pd.to_datetime(df.dtfaturamento,dayfirst=True, errors='ignore')
        cols_str = ['uf', 
                    'espessura',
                #   'grupo_materiais'
                    ]

        for col in cols_str:
            df[col] = df[col].astype(str)

        cols_fillna = ['regiao_2',
        'material',
        'familia',
        'grupo_analise' ,
        'cor',
        'espessura',
        'dimensao',
        'produto',
        'familia_repasse_1',
        'familia_repasse_2',
        'grupo_tipologico' ]

        for col in cols_fillna:
            df[col].fillna(col,inplace=True)

        cols_drop = ['tp_doc_faturamento']

        df.drop(columns = cols_drop,inplace=True)

        df['volume_t'] = pd.to_numeric(df['volume_t'], errors='coerce').fillna(0)

        aju_gt = df.groupby(['cliente_grupo','grupo_tipologico'])[['volume_t']].sum().sort_values(by=['volume_t'],ascending=False)
        aju_gt.reset_index(inplace=True)
        aju_gt.drop_duplicates(subset=['cliente_grupo'],keep='first', inplace=True)
        aju_gt.drop(columns=['volume_t'], inplace=True)
        df.drop(columns=['grupo_tipologico'], inplace=True)
        df = df.merge(aju_gt, on=['cliente_grupo'], how='left')
        df['cliente_grupo']  = df['cliente_grupo'] + '|' + df['gerente_contas'] 

        colunas_obj = ['dimensao']

        for col in colunas_obj:
            df[col] = df[col].astype(str)

        df = df[df.dtfaturamento.dt.to_period('M')<=mes_ref].copy()
        df.to_parquet(local_destino)

        print("Base com dados até ", df.dtfaturamento.max())
        print(f"check ETL: {df.shape[0]>50_000}" )