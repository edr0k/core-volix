import pandas as pd

# Dados de Teste (Cenário Realista)
dados = [
    # 1. Produto Estável e Sazonal (Cimento CP II)
    {"dt_ref": "2023-01-01", "vendas_kg": 5000, "faturamento": 25000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-02-01", "vendas_kg": 5200, "faturamento": 26000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-03-01", "vendas_kg": 5500, "faturamento": 27500, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-04-01", "vendas_kg": 5300, "faturamento": 26500, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-05-01", "vendas_kg": 6000, "faturamento": 30000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-06-01", "vendas_kg": 6200, "faturamento": 31000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-07-01", "vendas_kg": 6500, "faturamento": 32500, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-08-01", "vendas_kg": 6800, "faturamento": 34000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-09-01", "vendas_kg": 6700, "faturamento": 33500, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-10-01", "vendas_kg": 7000, "faturamento": 35000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-11-01", "vendas_kg": 7200, "faturamento": 36000, "familia": "Cimentos", "produto": "Cimento CP II"},
    {"dt_ref": "2023-12-01", "vendas_kg": 7500, "faturamento": 37500, "familia": "Cimentos", "produto": "Cimento CP II"},

    # 2. Produto com Tendência de Queda (Cimento Branco)
    {"dt_ref": "2023-01-01", "vendas_kg": 1200, "faturamento": 12000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-02-01", "vendas_kg": 1100, "faturamento": 11000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-03-01", "vendas_kg": 1300, "faturamento": 13000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-04-01", "vendas_kg": 1250, "faturamento": 12500, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-05-01", "vendas_kg": 1400, "faturamento": 14000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-06-01", "vendas_kg": 1350, "faturamento": 13500, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-07-01", "vendas_kg": 1500, "faturamento": 15000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-08-01", "vendas_kg": 1600, "faturamento": 16000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-09-01", "vendas_kg": 1550, "faturamento": 15500, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-10-01", "vendas_kg": 1700, "faturamento": 17000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-11-01", "vendas_kg": 1800, "faturamento": 18000, "familia": "Cimentos", "produto": "Cimento Branco"},
    {"dt_ref": "2023-12-01", "vendas_kg": 1900, "faturamento": 19000, "familia": "Cimentos", "produto": "Cimento Branco"},

    # 3. Produto de Baixo Volume (Argamassa)
    {"dt_ref": "2023-01-01", "vendas_kg": 300, "faturamento": 4500, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-02-01", "vendas_kg": 320, "faturamento": 4800, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-03-01", "vendas_kg": 310, "faturamento": 4650, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-04-01", "vendas_kg": 350, "faturamento": 5250, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-05-01", "vendas_kg": 340, "faturamento": 5100, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-06-01", "vendas_kg": 380, "faturamento": 5700, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-07-01", "vendas_kg": 400, "faturamento": 6000, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-08-01", "vendas_kg": 420, "faturamento": 6300, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-09-01", "vendas_kg": 410, "faturamento": 6150, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-10-01", "vendas_kg": 450, "faturamento": 6750, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-11-01", "vendas_kg": 460, "faturamento": 6900, "familia": "Argamassas", "produto": "Argamassa ACIII"},
    {"dt_ref": "2023-12-01", "vendas_kg": 500, "faturamento": 7500, "familia": "Argamassas", "produto": "Argamassa ACIII"},

    # 4. Produto Novo (Começou no meio do ano)
    {"dt_ref": "2023-07-01", "vendas_kg": 50, "faturamento": 1000, "familia": "Argamassas", "produto": "Rejunte"},
    {"dt_ref": "2023-08-01", "vendas_kg": 60, "faturamento": 1200, "familia": "Argamassas", "produto": "Rejunte"},
    {"dt_ref": "2023-09-01", "vendas_kg": 55, "faturamento": 1100, "familia": "Argamassas", "produto": "Rejunte"},
    {"dt_ref": "2023-10-01", "vendas_kg": 70, "faturamento": 1400, "familia": "Argamassas", "produto": "Rejunte"},
    {"dt_ref": "2023-11-01", "vendas_kg": 80, "faturamento": 1600, "familia": "Argamassas", "produto": "Rejunte"},
    {"dt_ref": "2023-12-01", "vendas_kg": 90, "faturamento": 1800, "familia": "Argamassas", "produto": "Rejunte"}
]

df = pd.DataFrame(dados)
df['dt_ref'] = pd.to_datetime(df['dt_ref'])

# Salva o arquivo Excel
df.to_excel("teste_api_vendas.xlsx", index=False)
print("Arquivo 'teste_api_vendas.xlsx' gerado com sucesso!")