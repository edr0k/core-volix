import requests
import json

# URL da sua API (verifique se a porta é 8000 mesmo)
url = "http://127.0.0.1:8000/v1/previsao/arquivo"

# Caminho do arquivo que criamos antes
caminho_arquivo = "teste_api_vendas.xlsx"

# PREPARAÇÃO DOS DADOS
# O segredo: converter os dicionários para STRING JSON usando json.dumps()
# Isso evita o erro 422 porque garante aspas duplas e formato correto.
config_json = json.dumps({
    "horizonte": 3,
    "n_clusters": 0,
    "regra_corte": True
})

map_json = json.dumps({
    "data": "dt_ref",
    "volume": "vendas_kg",
    "receita": "faturamento",
    "agrupamento": ["familia", "produto"]
})

# PAYLOAD (Campos do Formulário)
payload = {
    "config": config_json,
    "mapeamento": map_json,
    "formato_saida": "excel"
}

# ARQUIVO
try:
    with open(caminho_arquivo, 'rb') as f:
        arquivos = {
            'arquivo': (caminho_arquivo, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        
        print(f"Enviando requisição para {url}...")
        response = requests.post(url, data=payload, files=arquivos)

    # RESPOSTA
    if response.status_code == 200:
        print("✅ Sucesso! Baixando arquivo de resposta...")
        with open("resultado_previsao.xlsx", "wb") as f_out:
            f_out.write(response.content)
        print("Arquivo salvo como: resultado_previsao.xlsx")
        
    else:
        print(f"❌ Erro {response.status_code}:")
        print(response.text) # Aqui ele vai mostrar exatamente qual campo falhou

except FileNotFoundError:
    print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Gere o excel de teste primeiro.")
except Exception as e:
    print(f"Erro de conexão: {e}")