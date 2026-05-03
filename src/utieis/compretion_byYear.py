import tarfile
import os
import glob
import sys
from tqdm import tqdm
from pathlib import Path




def comprimir_arquivos_selecionados(arquivos, arquivo_tar_gz, diretorio_base='.'):
    """
    Comprime uma lista de arquivos em um arquivo tar.gz.

    Args:
        arquivos (list): Lista de caminhos para os arquivos a serem comprimidos.
        arquivo_tar_gz (str): Caminho para o arquivo tar.gz de saída.
        diretorio_base (str): Diretório base para os arquivos no arquivo tar.
    """
    try:
        with tarfile.open(os.path.join(arquivo_tar_gz), 'w:gz') as tar:
            for arquivo in tqdm(arquivos):
                # Garante que os caminhos sejam relativos ao diretório base
                # caminho_relativo = os.path.relpath(arquivo, diretorio_base)                
                tar.add(arquivo)

        print(f"Arquivos comprimidos com sucesso em '{arquivo_tar_gz}'.")
    except FileNotFoundError:
        print("Erro: Um ou mais arquivos não foram encontrados.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")


# Exemplo de uso:
pathInput = '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV/patches_pred_FotoVv2'
fileTAR = pathInput.split("/")[-1]    # Substitua pelo caminho do seu arquivo
pathparent = str(Path(pathInput).parents[0])
pathfileTar = os.path.join(pathparent,"filePredTAR")
# Cria o diretório de destino se ele não existir
if not os.path.exists(pathfileTar):
    os.makedirs(pathfileTar)

lstFilesnpy = glob.glob(pathInput + "/*.npy")
print(len(lstFilesnpy))
# sys.exit()
for nyear in range(2015, 2024):
    lstYY = []
    for nfile in tqdm(lstFilesnpy):    
        if str(nyear) in nfile:
            # print("adding file >> ", nfile)
            lstYY.append(os.path.join(pathInput, nfile))

    name_path_tar_year =  f"{fileTAR}_{nyear}.tar.gz"
    print(f" >>>>> saving file {name_path_tar_year} in directory {pathfileTar} ")    
    comprimir_arquivos_selecionados(lstYY, name_path_tar_year, pathfileTar)

