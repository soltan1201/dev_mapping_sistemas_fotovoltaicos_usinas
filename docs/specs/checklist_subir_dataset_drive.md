# Checklist: montar a base v4 e subir para o Drive para treinar

Script de upload: `src/preprocessing/subir_dataset_drive_rclone.py`

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

## B. Conta e remote do Drive

- [ ] A conta de destino é a do Colab (solkan.cengine17).
- [ ] O remote rclone usa login de **usuário**, não service account:
      ```bash
      rclone config update ee-solkaneengine17 service_account_file ""
      rclone config reconnect ee-solkaneengine17:
      # browser → n → rodar "rclone authorize drive ..." num PC com navegador,
      # logar em solkan.cengine17 e colar o token
      ```
- [ ] `rclone lsd ee-solkaneengine17: | head` lista as pastas do Meu Drive.
- [ ] Espaço livre no Drive maior que `du -sh ~/almacen/db_images/DS_FV_tfs_v4`
      (o script confere). Se faltar, apague a v3 do Drive, que já está dentro
      da v4.

## C. Upload

- [ ] `python src/preprocessing/subir_dataset_drive_rclone.py`, primeiro com
      **Dry-run = s**.
- [ ] Rodar de novo com **Dry-run = n**. Se cair, rodar de novo: ele só sobe o
      que falta.
- [ ] O fim mostra `0 differences found` e a tabela local × drive toda com ✅.

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
