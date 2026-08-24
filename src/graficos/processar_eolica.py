from pathlib import Path

import pandas as pd

DADOS_DIR = Path(__file__).parent
ARQUIVO_CONSOLIDADO = DADOS_DIR / "eolica_consolidado.csv"

# Nome bruto (sem espaço/acento, às vezes com erro de digitação) -> nome de
# exibição correto. Mesma convenção usada em processar_UFV.py.
NOMES_REGIAO = {
    # biomas
    "MataAtlantica": "Mata Atlântica",
    "caatinga": "Caatinga",
    "cerrado": "Cerrado",
    "pampa": "Pampa",
    # estados
    "Bahia": "Bahia",
    "ceara": "Ceará",
    "maranhao": "Maranhão",
    "paraiba": "Paraíba",
    "pernambuco": "Pernambuco",
    "piaui": "Piauí",
    "rioGrandeNorte": "Rio Grande do Norte",
    "rioGrandeSul": "Rio Grande do Sul",
    "rioJaneiro": "Rio de Janeiro",
    "santaCatalina": "Santa Catarina",  # corrige erro de digitação do arquivo original
    "sergipe": "Sergipe",
    # país
    "Brasil": "Brasil",
}

# Nome de exibição -> sigla. Estados usam a sigla oficial (UF); biomas e país
# não têm sigla oficial, usa-se uma abreviação de 3 letras por convenção.
SIGLAS_REGIAO = {
    # biomas
    "Mata Atlântica": "MAT",
    "Caatinga": "CAT",
    "Cerrado": "CER",
    "Pampa": "PAM",
    # estados (UF oficial)
    "Bahia": "BA",
    "Ceará": "CE",
    "Maranhão": "MA",
    "Paraíba": "PB",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rio de Janeiro": "RJ",
    "Santa Catarina": "SC",
    "Sergipe": "SE",
    # país
    "Brasil": "BR",
}


def processar_consolidado(arquivo: Path) -> pd.DataFrame:
    df = pd.read_csv(arquivo)
    df["nome_region"] = df["nome_region"].map(lambda n: NOMES_REGIAO.get(n, n))
    df["sigla_region"] = df["nome_region"].map(SIGLAS_REGIAO)
    return df


if __name__ == "__main__":
    resultado = processar_consolidado(ARQUIVO_CONSOLIDADO)
    print(resultado.head(10))
    print(resultado.tail(10))
    resultado.to_csv(ARQUIVO_CONSOLIDADO, index=False)
