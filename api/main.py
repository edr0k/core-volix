from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import Json
import pandas as pd
import io
import logging

# Importa seus schemas e serviços já criados
from .schemas import PayloadEntrada, ConfigPrevisao, ColunasMapping
from .services import PrevisaoService

app = FastAPI(title="Core Volix API - File Engine")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/v1/previsao/arquivo")
async def processar_arquivo(
    # 1. O Arquivo (Excel ou Parquet)
    arquivo: UploadFile = File(..., description="Arquivo histórico (.xlsx ou .parquet)"),
    
    # 2. As Configurações (Vêm como JSON string dentro do formulário)
    # No Swagger, aparecerá um campo texto para colar o JSON da config
    config: Json[ConfigPrevisao] = Form(..., description='JSON de configuração: {"horizonte": 3, "regra_corte": true}'),
    mapeamento: Json[ColunasMapping] = Form(..., description='JSON de mapeamento: {"data": "dt_ref", "volume": "qtd", "agrupamento": ["produto"]}'),
    
    # 3. Formato de Saída (Opcional)
    formato_saida: str = Form("excel", enum=["excel", "parquet"], description="Formato do arquivo de resposta")
):
    """
    Recebe um arquivo (Excel/Parquet) e retorna as previsões no formato solicitado.
    """
    try:
        logger.info(f"Recebendo arquivo: {arquivo.filename}")

        # --- A. LER O ARQUIVO ---
        # Detecta a extensão para usar o leitor correto do Pandas
        filename = arquivo.filename.lower()
        contents = await arquivo.read()
        buffer_entrada = io.BytesIO(contents)

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df_entrada = pd.read_excel(buffer_entrada)
        elif filename.endswith(".parquet"):
            df_entrada = pd.read_parquet(buffer_entrada)
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use .xlsx ou .parquet")

        if df_entrada.empty:
            raise HTTPException(status_code=400, detail="O arquivo enviado está vazio.")

        # --- B. PROCESSAR (Usa o mesmo Service de antes!) ---
        servico = PrevisaoService(
            df=df_entrada,
            mapeamento=mapeamento,
            config=config
        )
        
        df_resultado = servico.executar()

        if df_resultado.empty:
            return {"status": "warning", "message": "Nenhuma previsão gerada (dados insuficientes)."}

        # --- C. GERAR ARQUIVO DE SAÍDA ---
        buffer_saida = io.BytesIO()

        if formato_saida == "excel":
            # Salva como Excel em memória
            with pd.ExcelWriter(buffer_saida, engine='xlsxwriter') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Previsao')
            
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename_out = "previsao_resultado.xlsx"
            
        else: # Parquet
            df_resultado.to_parquet(buffer_saida, index=False)
            media_type = "application/octet-stream"
            filename_out = "previsao_resultado.parquet"

        buffer_saida.seek(0) # Volta o ponteiro para o início do arquivo

        # Retorna o arquivo como download
        return StreamingResponse(
            buffer_saida,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename_out}"}
        )

    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))