# Checklist: montar a base v4 e subir para o Drive para treinar

Script de upload: `src/preprocessing/subir_dataset_drive.py` (Drive API + token do
gcloud, sem rclone)

## A. Base pronta no servidor (I7)

- [ ] Download da v3 completo: `train` 600, `val` 130, `test` 128 arquivos em
      `~/almacen/db_images/DS_FV_tfs_scaled_v3`, sem `*.part`.
- [ ] Amostras novas baixadas em `~/almacen/db_images/tfrecord_fv_revisao`.
- [ ] Espaço em disco suficiente: `df -h ~/almacen/db_images` (cerca de 2× o
      tamanho da v3 + novas).
- [ ] `remontar_dataset_tfrecord.py` rodado com **Dry-run = n**.
- [ ] Existe `~/almacen/db_images/DS_FV_tfs_v4/manifesto.json`, e
      `_baldes_tmp/` não existe mais.
- [ ] (opcional) `src/dash/dashboard_tfrecord.py --root ~/almacen/db_images/DS_FV_tfs_v4`:
      conferir os patches com e sem FV por split e ver algumas imagens.

## B. Conta do Drive (gcloud)

- [ ] A conta de destino é a do Colab (solkan.cengine17).
- [ ] Login no gcloud com acesso ao Drive (o `--force` é obrigatório se a conta
      já estava logada sem esse acesso):
      ```bash
      gcloud auth login solkan.cengine17@gmail.com --enable-gdrive-access --no-launch-browser --force
      # abrir a URL num navegador, logar nessa conta, aceitar o Drive, colar o código
      gcloud auth list      # a conta aparece na lista
      ```
- [ ] Espaço livre no Drive maior que o dataset (cerca de 20,3 GB). A conta
      grátis tem 15 GB no total. O script mostra o espaço livre; se faltar,
      apague a v3 do Drive (`DS_FV_tfs_scaled_v3`) e esvazie a lixeira, ou use
      uma conta com mais armazenamento.

## C. Upload

- [ ] `python src/preprocessing/subir_dataset_drive.py`, primeiro com
      **Dry-run = s**: confere a conta conectada, o espaço e o tamanho local.
- [ ] Rodar de novo com **Dry-run = n**. Se cair, rodar de novo: ele só sobe o
      que falta.
- [ ] O fim mostra a tabela local × drive toda com ✅ e "Upload completo e
      conferido".

## D. Treino no Colab (`train_fotovoltaica_colab_v2.ipynb`)

- [ ] Célula 12:
      ```python
      DATASET_ROOT_DS = '/content/drive/MyDrive/DS_FV_tfs_v4'
      OUTPUT_DIR      = '/content/drive/MyDrive/TRAINING_OUTPUTS_FV_V4'
      ```
- [ ] (recomendado) Copiar para o disco local do Colab e apontar para lá:
      ```python
      !rsync -a --info=progress2 /content/drive/MyDrive/DS_FV_tfs_v4/ /content/DS_FV_tfs_v4/
      DATASET_ROOT_DS = '/content/DS_FV_tfs_v4'
      ```
- [ ] O nº de arquivos por split impresso na célula 12 bate com o do servidor.
- [ ] `show_batch` mostra patches com rótulo coerente.
- [ ] Rever `EMPTY_PATCH_KEEP_RATE` (0.25) com a nova proporção de patches sem
      FV mostrada no dashboard.
- [ ] Anotar no nome ou histórico do modelo que ele foi treinado com a v4.

## E. Depois do treino

- [ ] Rodar o predict do modelo novo nos TIFs de omissão:
      `makePredict_fromTIF.py --model-path <modelo> --input-dir ~/almacen/db_images/tif_fv_omissao`.
- [ ] Comparar com o modelo da v3 nos mesmos pontos.
