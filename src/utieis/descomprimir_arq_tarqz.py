import tarfile
import os

def descomprimir_tar_gz(arquivo_tar_gz, diretorio_TAR, diretorio_destino='.'):
    """
    Descomprime um arquivo tar.gz para um diretório de destino.

    Args:
        arquivo_tar_gz (str): Caminho para o arquivo tar.gz.
        diretorio_destino (str): Caminho para o diretório onde os arquivos serão extraídos.
                                  O padrão é o diretório atual.
    """
    try:
        with tarfile.open(os.path.join(diretorio_TAR, arquivo_tar_gz), 'r:gz') as tar:
            tar.extractall(path=diretorio_destino)
        print(f"Arquivo '{arquivo_tar_gz}' descompactado com sucesso em '{diretorio_destino}'.")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_tar_gz}' não encontrado.")
    except tarfile.ReadError:
        print(f"Erro: Arquivo '{arquivo_tar_gz}' não é um arquivo tar.gz válido.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Exemplo de uso:
fileTAR = 'patches_FotoVv2y.tar.gz' # Substitua pelo caminho do seu arquivo
arquivo = input("Digit the name of file tar.gz: ")
if str(fileTAR) != str(arquivo) and arquivo:
    fileTAR = arquivo

pathInput = '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV'
partida = input("Digit the path where the file tar.gz is : ")

if str(pathInput) != str(partida) and partida:
    pathInput = partida

pathDestination = '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV' # Substitua pelo diretório de destino desejado
destino = input("Digit the path of file tar.gz wil be descompact: ")
if str(pathDestination) != str(destino) and destino:
    pathDestination = destino

# Cria o diretório de destino se ele não existir
if not os.path.exists(pathDestination):
    os.makedirs(pathDestination)

descomprimir_tar_gz(fileTAR, pathInput, pathDestination)