from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# 1. Mapeamento de Colunas
# O cliente diz: "A minha coluna de data se chama 'DT_VENDA'"
class ColunasMapping(BaseModel):
    data: str = Field(..., description="Nome da coluna de data (ex: 'dtfaturamento')")
    volume: str = Field(..., description="Nome da coluna de volume (ex: 'volume_t')")
    receita: str = Field(None, description="Nome da coluna de receita (opcional para volume, obrigatório para elasticidade)")
    agrupamento: List[str] = Field(..., description="Lista de colunas para agrupar (ex: ['cod_cliente', 'familia'])")

# 2. Configurações da Previsão
# O cliente diz: "Quero prever 6 meses para frente"
class ConfigPrevisao(BaseModel):
    horizonte: int = Field(3, description="Quantos meses prever para frente")
    n_clusters: int = Field(7, description="Número de clusters para perfil (se usar)")
    regra_corte: bool = Field(True, description="Se deve aplicar corte de meta vs previsão")

# 3. O Pacote Completo (Payload)
# É isso que o cliente envia no POST
class PayloadEntrada(BaseModel):
    config: ConfigPrevisao
    mapeamento: ColunasMapping
    dados: List[Dict[str, Any]] # A lista de dicionários (o JSON dos dados)