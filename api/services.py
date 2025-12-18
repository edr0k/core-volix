import pandas as pd
from prophet import Prophet
from .schemas import ConfigPrevisao, ColunasMapping
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrevisaoService:
    def __init__(self, df: pd.DataFrame, mapeamento: ColunasMapping, config: ConfigPrevisao):
        self.df = df
        self.map = mapeamento
        self.config = config
        
    def _preparar_dados(self):
        """
        Traduz as colunas do cliente para o padrão interno (ds, y).
        """
        rename_dict = {
            self.map.data: 'ds',
            self.map.volume: 'y'
        }
        
        cols_necessarias = [self.map.data, self.map.volume] + self.map.agrupamento
        
        # Filtra e renomeia
        df_prep = self.df[cols_necessarias].copy()
        df_prep.rename(columns=rename_dict, inplace=True)
        
        # --- CORREÇÃO DE DATA ---
        # Garante que o Pandas entenda datas brasileiras (dia/mês) e ignore erros
        df_prep['ds'] = pd.to_datetime(df_prep['ds'], dayfirst=True, errors='coerce')
        df_prep = df_prep.dropna(subset=['ds'])
        
        return df_prep

    def executar(self):
        df_trabalho = self._preparar_dados()
        resultados = []
        
        grupos = self.map.agrupamento
        
        if not grupos:
            iterador = [('Total', df_trabalho)]
        else:
            iterador = df_trabalho.groupby(grupos)

        # Evita logar se o iterador estiver vazio
        if not df_trabalho.empty:
            logger.info(f"Processando grupos...")

        for chaves, df_grupo in iterador:
            
            # Pula grupos sem dados suficientes para prever
            if len(df_grupo) < 2:
                continue 

            model = Prophet()
            
            try:
                model.fit(df_grupo)
                
                future = model.make_future_dataframe(periods=self.config.horizonte, freq='MS')
                forecast = model.predict(future)
                
                # Pega apenas os meses futuros
                res = forecast[['ds', 'yhat']].tail(self.config.horizonte).copy()
                
                # --- CORREÇÃO DO ERRO 'Length of values' ---
                if grupos:
                    # Caso 1: Agrupamento único (Ex: ['produto'])
                    if len(grupos) == 1:
                        # Se o Pandas devolveu uma tupla ('A',), pegamos só o 'A'
                        # Isso garante que ele faça o "broadcast" (repita o valor nas 3 linhas)
                        valor = chaves[0] if isinstance(chaves, tuple) else chaves
                        res[grupos[0]] = valor
                    
                    # Caso 2: Agrupamento múltiplo (Ex: ['cliente', 'familia'])
                    else:
                        for i, col in enumerate(grupos):
                            res[col] = chaves[i]
                
                resultados.append(res)
                
            except Exception as e:
                # Loga o erro mas não derruba a API inteira
                logger.error(f"Erro no grupo {chaves}: {e}")
                continue

        if not resultados:
            return pd.DataFrame()

        return pd.concat(resultados)