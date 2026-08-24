import re
from pathlib import Path

import pandas as pd

DADOS_DIR = Path(__file__).parent / "dados"
VERSION = "0-4-12-spt-4"
# Alguns arquivos (Pampa, Espírito Santo, Paraná, Rio Grande do Sul, Santa
# Catarina) nunca tiveram a série de versões "0-4-*" processada — só têm a
# versão da coleção oficial. Sem fallback, esses arquivos são filtrados por
# completo e a região some do CSV consolidado.
VERSION_FALLBACK = "collection_10_1"
PADRAO_NOME = re.compile(r"^usina_FV_(?P<tipo_region>[^_]+)_(?P<nome_region>.+)\.csv$")

# Nome bruto extraído do arquivo (sem espaço/acento, às vezes com erro de
# digitação) -> nome de exibição correto.
NOMES_REGIAO = {
    # biomas
    "Amazonia": "Amazônia",
    "Caatinga": "Caatinga",
    "Cerrado": "Cerrado",
    "MataAtlantica": "Mata Atlântica",
    "Pampa": "Pampa",
    # estados
    "Bahia": "Bahia",
    "Ceara": "Ceará",
    "EspiritoSanto": "Espírito Santo",
    "MinasGerais": "Minas Gerais",
    "Paraiba": "Paraíba",
    "Parana": "Paraná",
    "Pernambuco": "Pernambuco",
    "Piaui": "Piauí",
    "RioGrandeNorte": "Rio Grande do Norte",
    "RioGrandeSul": "Rio Grande do Sul",
    "Rondonia": "Rondônia",
    "SantaCatalina": "Santa Catarina",  # corrige erro de digitação do arquivo original
    "SaoPaulo": "São Paulo",
    "Tocantins": "Tocantins",
    # país
    "Br": "Brasil",
}

# Nome de exibição -> sigla. Estados usam a sigla oficial (UF); biomas e país
# não têm sigla oficial, usa-se uma abreviação de 3 letras por convenção.
SIGLAS_REGIAO = {
    # biomas
    "Amazônia": "AMZ",
    "Caatinga": "CAT",
    "Cerrado": "CER",
    "Mata Atlântica": "MAT",
    "Pampa": "PAM",
    # estados (UF oficial)
    "Bahia": "BA",
    "Ceará": "CE",
    "Espírito Santo": "ES",
    "Minas Gerais": "MG",
    "Paraíba": "PB",
    "Paraná": "PR",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rondônia": "RO",
    "Santa Catarina": "SC",
    "São Paulo": "SP",
    "Tocantins": "TO",
    # país
    "Brasil": "BR",
}


def processar_arquivos(dados_dir: Path, version: str, version_fallback: str | None = None) -> pd.DataFrame:
    dfs = []
    for arquivo in sorted(dados_dir.glob("usina_FV_*.csv")):
        match = PADRAO_NOME.match(arquivo.name)
        if not match:
            continue

        df_arquivo = pd.read_csv(arquivo)
        versoes_disponiveis = df_arquivo["version"].unique()

        versao_usada = version
        if versao_usada not in versoes_disponiveis:
            if version_fallback is not None and version_fallback in versoes_disponiveis:
                print(f"  aviso: {arquivo.name} não tem a versão '{version}' — "
                      f"usando fallback '{version_fallback}'")
                versao_usada = version_fallback
            else:
                print(f"  aviso: {arquivo.name} não tem a versão '{version}' nem "
                      f"'{version_fallback}' — arquivo ignorado")
                continue

        df = df_arquivo[df_arquivo["version"] == versao_usada].copy()
        df["tipo_region"] = match.group("tipo_region")
        nome_bruto = match.group("nome_region")
        df["nome_region"] = NOMES_REGIAO.get(nome_bruto, nome_bruto)
        df["sigla_region"] = df["nome_region"].map(SIGLAS_REGIAO)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    resultado = processar_arquivos(DADOS_DIR, VERSION, VERSION_FALLBACK)
    print(resultado.head(10))
    print(resultado.tail(10))
    resultado.to_csv(DADOS_DIR.parent / "UFV_consolidado.csv", index=False)
