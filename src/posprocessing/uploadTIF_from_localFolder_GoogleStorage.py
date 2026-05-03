from google.cloud import storage
from io import BytesIO
# import pandas as pdw
import glob
import os

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage using a service account key.
    
    Args:
        json_key_path (str): Path to the JSON key file.
        project_name (str): Google Cloud project name.
        bucket_name (str): Name of the GCS bucket.
        source_file_name (str): Path to the local file to upload.
        destination_blob_name (str): Name of the object in the bucket.
    """
    _json_key_path = '/home/superuser/Dados/mapbiomas/mykeys/mapbiomas-agua-36521f541610.json'
    _project_name = 'mapbiomas-agua'

    # Initialize a storage client using the service account
    storage_client = storage.Client.from_service_account_json(_json_key_path, project=_project_name)

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object (path in the bucket)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} \n >>>>>>>>> uploaded to {destination_blob_name}")

# Example usage
bucket_name = "mapbiomas-energia"
# path_base = '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/imgRegions_FotoV/predTIF_FotoVv2'
# path_base = '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/patches_pred_FotoV/'
path_base = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_pred_tif_g2C'
# path_baseChange = '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/imgRegions_FotoV/gdal_predTIF_FotoVv2/'
path_baseChange = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/gdal_array_pred_tif_g2C_corr'
source_file_name = "./myfile.txt"
destination_blob_name = "fotovoltaicas_tif"

for nyear in range(2015, 2025):
    print("=============================================================")
    lstpathTIF = glob.glob(os.path.join(path_base, str(nyear)) + '/*.tif')
    print(" loading tif in >>>>> \n >> ", os.path.join(path_base, str(nyear)))
    print("=============================================================")
    # Cria o diretório de destino se ele não existir
    pathDest = os.path.join(path_baseChange, str(nyear))
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)

    for cc, namepath in enumerate(lstpathTIF[:]):
        if  cc  > -1:
            source_file_name = namepath
            nameFileTIF = source_file_name.replace(os.path.join(path_base, str(nyear)) + '/', "")
            # change configuration TIF compressao The required TIFF tag 'TileWidth' is not present in the IFD at index 0
            # https://github.com/cogeotiff/cog-spec/blob/master/spec.md
            source_tif_name = os.path.join(pathDest, nameFileTIF)
            comandoOS = f"gdal_translate {source_file_name} {source_tif_name} -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co COMPRESS=LZW"
            os.system(comandoOS)
            ################################################################
            print(f" # {cc}   > ...{source_file_name.replace(path_base, "")} ") 
            destination_blob_name = f"fotovoltaicas_tif/{nameFileTIF.replace('.tif', '_g2d.tif')}"
            print(" destination >>> ", destination_blob_name)
            print("  ")
            upload_to_gcs(bucket_name, source_tif_name, destination_blob_name)