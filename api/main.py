import json  # <--- IMPORTANTE: Adicione este import no topo
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import logging

# Importa seus schemas e serviços
from .schemas import PayloadEntrada, ConfigPrevisao, ColunasMapping
from .services import PrevisaoService

app = FastAPI(title="Core Volix API - File Engine")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/v1/previsao/arquivo")
async def processar_arquivo(
    # 1. O Arquivo
    arquivo: UploadFile = File(..., description="Arquivo histórico (.xlsx ou .parquet)"),
    
    # 2. As Configurações (Recebemos como TEXTO PURO para evitar erro 422)
    config: str = Form(..., description='JSON string de configuração'),
    mapeamento: str = Form(..., description='JSON string de mapeamento'),
    
    # 3. Formato de Saída
    formato_saida: str = Form("excel", enum=["excel", "parquet"], description="Formato do arquivo de resposta")
):
    """
    Recebe um arquivo (Excel/Parquet) e retorna as previsões.
    """
    try:
        # --- A. PARSE MANUAL DOS JSONs (A Mágica da correção) ---
        # Convertemos a string que veio do formulário em dicionário Python
        try:
            config_dict = json.loads(config)
            map_dict = json.loads(mapeamento)
            
            # Valida os dados usando seus Schemas Pydantic
            config_obj = ConfigPrevisao(**config_dict)
            map_obj = ColunasMapping(**map_dict)
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="O JSON de configuração ou mapeamento está inválido (erro de sintaxe).")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Erro de validação nos parâmetros: {str(e)}")

        logger.info(f"Recebendo arquivo: {arquivo.filename}")

        # --- B. LER O ARQUIVO ---
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

        # --- C. PROCESSAR ---
        servico = PrevisaoService(
            df=df_entrada,
            mapeamento=map_obj,   # Passamos o objeto já validado
            config=config_obj     # Passamos o objeto já validado
        )
        
        df_resultado = servico.executar()

        if df_resultado.empty:
            return {"status": "warning", "message": "Nenhuma previsão gerada (dados insuficientes)."}

        # --- D. GERAR ARQUIVO DE SAÍDA ---
        buffer_saida = io.BytesIO()

        if formato_saida == "excel":
            with pd.ExcelWriter(buffer_saida, engine='xlsxwriter') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Previsao')
            
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename_out = "previsao_resultado.xlsx"
            
        else: # Parquet
            df_resultado.to_parquet(buffer_saida, index=False)
            media_type = "application/octet-stream"
            filename_out = "previsao_resultado.parquet"

        buffer_saida.seek(0)

        return StreamingResponse(
            buffer_saida,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename_out}"}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))