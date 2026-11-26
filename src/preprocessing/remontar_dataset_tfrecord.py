#!/usr/bin/env python3
"""
Remonta o dataset de TFRecords: base antiga (v3) + amostras novas de revisão
=============================================================================
Junta
  - DS_FV_tfs_scaled_v3/{train,val,test}/*.tfrecord.gz   (baixado do Drive)
  - tfrecord_fv_revisao/tfrecord_fvrev_*.tfrecord.gz     (download_dataset_tfrecord_fv_revisao.py)
embaralha no nível de REGISTRO (patch, não arquivo) e grava um dataset novo
  <saída>/{train,val,test}/tfrecord_fv_<tag>_<split>_partNNN.tfrecord.gz
no mesmo formato que train_fotovoltaica_colab_v2.ipynb lê
(bandas blue, green, red, pvi, pvpi + label; float_list 257x257; GZIP).

Divisão train/val/test
  - v3: por padrão cada registro FICA no split em que já estava (o split da v3
    foi feito por região; misturar geraria vazamento). Opção "refazer" junta
    tudo e redivide por arquivo.
  - revisão: patches vizinhos se sobrepõem (vários patches por polígono), então
    a divisão é por BLOCO espacial de BLOCO_KM km: todos os patches de um bloco
    vão para o mesmo split. As coordenadas vêm de
        1. <shard>.pontos.csv (sidecar gravado pelo download, mesma ordem), ou
        2. pontos_<tipo>_<ano>_<versao>.csv, se o nº de linhas do shard bater
           com o nº de registros do arquivo;
    sem coordenadas, o arquivo inteiro vira um grupo.

Embaralhamento (sem carregar tudo na memória, ~1,6 MB por patch)
  passo 1: lê cada registro, valida, e escreve num "balde" aleatório do seu
           split (arquivos temporários GZIP);
  passo 2: cada balde é carregado, embaralhado e gravado como arquivo final.
  Cada arquivo final fica com ~REGISTROS_POR_ARQUIVO patches.

Registros inválidos (falta banda, tamanho != 257x257) são descartados e
contados. No fim grava <saída>/manifesto.json com as contagens.

Uso:
    python src/preprocessing/remontar_dataset_tfrecord.py
    Parâmetros pedidos via input(); Enter aceita o [padrão].
"""

import csv
import json
import math
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ==============================================================================
# 1. CONFIGURAÇÕES (padrões das perguntas)
# ==============================================================================
DIR_V3        = '~/almacen/db_images/DS_FV_tfs_scaled_v3'
DIR_REVISAO   = '~/almacen/db_images/tfrecord_fv_revisao'
DIR_SAIDA     = '~/almacen/db_images/DS_FV_tfs_v4'
TAG           = 'v4'
SPLITS        = ['train', 'val', 'test']
RATIOS_NOVOS  = (0.8, 0.1, 0.1)
BLOCO_KM      = 5.0
REG_POR_ARQ   = 100
SEED          = 42

BANDS      = ['blue', 'green', 'red', 'pvi', 'pvpi']
LABEL      = 'label'
KEYS       = BANDS + [LABEL]
RAW        = 257
N_VALORES  = RAW * RAW
R_MERCATOR = 6378137.0

# ==============================================================================
# 2. ENTRADA
# ==============================================================================

def ask(pergunta, padrao, cast=str, opcoes=None):
    while True:
        txt = input(f'{pergunta} [{padrao}]: ').strip()
        if not txt:
            return padrao
        try:
            val = cast(txt)
        except ValueError:
            print(f'  valor inválido: {txt}')
            continue
        if opcoes and val not in opcoes:
            print(f'  opções válidas: {opcoes}')
            continue
        return val


def ask_sim_nao(pergunta, padrao=False):
    txt = input(f'{pergunta} (s/n) [{"s" if padrao else "n"}]: ').strip().lower()
    return padrao if not txt else txt in ('s', 'sim', 'y', 'yes')


def ask_ratios(padrao):
    while True:
        txt = input(f'Proporção train val test das amostras novas [{" ".join(map(str, padrao))}]: ').strip()
        if not txt:
            return padrao
        try:
            r = tuple(float(v) for v in txt.split())
            if len(r) == 3 and abs(sum(r) - 1) < 1e-6:
                return r
        except ValueError:
            pass
        print('  informe 3 números que somem 1 (ex.: 0.8 0.1 0.1)')

# ==============================================================================
# 3. FONTES
# ==============================================================================

def listar_v3(raiz):
    """[(arquivo, split)] da base v3."""
    out = []
    for split in SPLITS:
        for f in sorted((raiz / split).glob('*.tfrecord.gz')):
            out.append((f, split))
    return out


_REV_RE = re.compile(r'^tfrecord_fvrev_(?P<tipo>[a-z]+)_(?P<ano>\d{4})_(?P<versao>.+)_part(?P<shard>\d+)\.tfrecord\.gz$')


def coords_revisao(arq, n_registros):
    """lon/lat de cada registro do shard (mesma ordem) ou None."""
    sidecar = arq.with_name(arq.name.replace('.tfrecord.gz', '.pontos.csv'))
    if sidecar.exists():
        with open(sidecar, encoding='utf-8') as fh:
            rows = [(float(r['lon']), float(r['lat'])) for r in csv.DictReader(fh)]
        if len(rows) == n_registros:
            return rows
    m = _REV_RE.match(arq.name)
    if m:
        idx = arq.parent / f'pontos_{m["tipo"]}_{m["ano"]}_{m["versao"]}.csv'
        if idx.exists():
            shard = int(m['shard'])
            with open(idx, encoding='utf-8') as fh:
                rows = [(float(r['lon']), float(r['lat'])) for r in csv.DictReader(fh)
                        if int(r['shard']) == shard]
            if len(rows) == n_registros:
                return rows
    return None


def bloco(lon, lat, bloco_m):
    x = R_MERCATOR * math.radians(lon)
    y = R_MERCATOR * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return f'{math.floor(x / bloco_m)}_{math.floor(y / bloco_m)}'

# ==============================================================================
# 4. LEITURA / VALIDAÇÃO
# ==============================================================================

def ler_registros(tf, arq):
    return list(tf.data.TFRecordDataset(str(arq), compression_type='GZIP').as_numpy_iterator())


def valida(tf, raw):
    """(ok, tem_fv) — confere bandas e tamanho 257x257."""
    try:
        ex = tf.train.Example.FromString(raw)
    except Exception:
        return False, False
    feat = ex.features.feature
    for k in KEYS:
        if k not in feat or len(feat[k].float_list.value) != N_VALORES:
            return False, False
    lab = np.asarray(feat[LABEL].float_list.value, dtype=np.float32)
    return True, bool(np.nansum(lab) > 0)


def contar_registros(tf, arqs):
    return sum(sum(1 for _ in tf.data.TFRecordDataset(str(a), compression_type='GZIP'))
               for a in arqs)

# ==============================================================================
# 5. DIVISÃO DAS AMOSTRAS NOVAS
# ==============================================================================

def dividir_grupos(tamanhos, ratios, seed):
    """{grupo: n} → {grupo: split}, preenchendo os splits por nº de registros."""
    grupos = list(tamanhos)
    random.Random(seed).shuffle(grupos)
    total = sum(tamanhos.values())
    alvo  = {s: r * total for s, r in zip(SPLITS, ratios)}
    atual = Counter()
    saida = {}
    for g in grupos:
        # split com maior déficit relativo
        s = max(SPLITS, key=lambda s: (alvo[s] - atual[s]) / (alvo[s] or 1))
        saida[g] = s
        atual[s] += tamanhos[g]
    return saida

# ==============================================================================
# 6. PIPELINE
# ==============================================================================

def main():
    print('=' * 60)
    print('Parâmetros de entrada (Enter = padrão)')
    print('=' * 60)
    dir_v3   = Path(ask('Pasta da base v3 (com train/ val/ test/)', DIR_V3)).expanduser()
    dir_rev  = Path(ask('Pasta das amostras novas (revisão)', DIR_REVISAO)).expanduser()
    dir_out  = Path(ask('Pasta de saída do dataset novo', DIR_SAIDA)).expanduser()
    tag      = ask('Tag do nome dos arquivos', TAG)
    modo_v3  = ask('Split da v3 (manter | refazer)', 'manter', opcoes=['manter', 'refazer'])
    ratios   = ask_ratios(RATIOS_NOVOS)
    bloco_km = ask('Tamanho do bloco espacial (km) das amostras novas', BLOCO_KM, float)
    reg_arq  = ask('Registros por arquivo de saída', REG_POR_ARQ, int)
    seed     = ask('Semente', SEED, int)
    dry_run  = ask_sim_nao('Dry-run (só contar e mostrar a divisão)?', False)

    arqs_v3  = listar_v3(dir_v3) if dir_v3.exists() else []
    arqs_rev = sorted(dir_rev.glob('*.tfrecord.gz')) if dir_rev.exists() else []
    print(f'\nv3      : {len(arqs_v3)} arquivos  ({dir_v3})')
    for s in SPLITS:
        print(f'  {s}: {sum(1 for _, sp in arqs_v3 if sp == s)}')
    print(f'revisão : {len(arqs_rev)} arquivos  ({dir_rev})')
    if not arqs_v3 and not arqs_rev:
        sys.exit('❌ Nenhum TFRecord encontrado.')

    if dir_out.exists() and any(dir_out.glob('*/*.tfrecord.gz')):
        sys.exit(f'❌ {dir_out} já tem TFRecords. Use outra pasta de saída (não sobrescrevo).')

    print(f'\nResumo: saída={dir_out} | tag={tag} | v3={modo_v3} | novos={ratios} '
          f'| bloco={bloco_km} km | {reg_arq} reg/arquivo | seed={seed} | dry_run={dry_run}')
    if not ask_sim_nao('Confirmar e continuar?', True):
        return

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    # --- grupos das amostras novas (por bloco espacial) -----------------------
    print('\nContando registros das amostras novas...', flush=True)
    rev_info = []           # (arq, [grupo por registro])
    tam_grupo = Counter()
    sem_coord = 0
    for arq in arqs_rev:
        n = contar_registros(tf, [arq])
        coords = coords_revisao(arq, n)
        if coords is None:
            sem_coord += 1
            grupos = [f'arq:{arq.name}'] * n
        else:
            grupos = [f'bloco:{bloco(lon, lat, bloco_km * 1000)}' for lon, lat in coords]
        tam_grupo.update(grupos)
        rev_info.append((arq, grupos))
    split_grupo = dividir_grupos(tam_grupo, ratios, seed) if tam_grupo else {}
    if sem_coord:
        print(f'  ⚠️  {sem_coord} arquivo(s) sem coordenadas: o arquivo inteiro vira um grupo.')

    # --- v3 -------------------------------------------------------------------
    if modo_v3 == 'refazer':
        # redivide por arquivo (split original ignorado)
        tam_v3 = {f'v3:{a.name}': 1 for a, _ in arqs_v3}
        sp = dividir_grupos(tam_v3, ratios, seed + 1)
        arqs_v3 = [(a, sp[f'v3:{a.name}']) for a, _ in arqs_v3]

    # --- estimativa de registros por split (para nº de baldes) -----------------
    amostra = [a for a, _ in arqs_v3[:5]]
    media_v3 = (contar_registros(tf, amostra) / len(amostra)) if amostra else 0
    est = Counter()
    for _, s in arqs_v3:
        est[s] += media_v3
    for arq, grupos in rev_info:
        for g in grupos:
            est[split_grupo[g]] += 1
    print('\nEstimativa de registros por split:')
    for s in SPLITS:
        print(f'  {s}: ~{int(est[s])}')
    if dry_run:
        n_rev = Counter(split_grupo[g] for _, gs in rev_info for g in gs)
        print('Amostras novas por split (exato):', dict(n_rev))
        print(f'Grupos espaciais das amostras novas: {len(tam_grupo)}')
        return

    # --- passo 1: espalhar em baldes ------------------------------------------
    tmp_dir = dir_out / '_baldes_tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    opts  = tf.io.TFRecordOptions(compression_type='GZIP')
    rng   = random.Random(seed)
    n_bal = {s: max(1, math.ceil(est[s] / reg_arq)) for s in SPLITS if est[s] > 0}
    writers = {s: [tf.io.TFRecordWriter(str(tmp_dir / f'{s}_{k:05d}.gz'), options=opts)
                   for k in range(n)] for s, n in n_bal.items()}

    stats = defaultdict(Counter)   # stats[split][fonte|fonte_fv|invalido]
    tarefas = [(a, s, None, 'v3') for a, s in arqs_v3] + \
              [(a, None, g, 'revisao') for a, g in rev_info]
    t0 = time.time()
    print(f'\nPasso 1/2: lendo {len(tarefas)} arquivos e espalhando em baldes...')
    for i, (arq, split, grupos, fonte) in enumerate(tarefas, 1):
        try:
            registros = ler_registros(tf, arq)
        except Exception as exc:
            print(f'\n  ❌ {arq.name}: {exc}')
            stats['erro_arquivo'][fonte] += 1
            continue
        for k, raw in enumerate(registros):
            s = split if grupos is None else split_grupo[grupos[k]]
            ok, tem_fv = valida(tf, raw)
            if not ok:
                stats[s]['invalido'] += 1
                continue
            if s not in writers:            # split sem estimativa (raro)
                n_bal[s] = 1
                writers[s] = [tf.io.TFRecordWriter(str(tmp_dir / f'{s}_00000.gz'), options=opts)]
            rng.choice(writers[s]).write(raw)
            stats[s][fonte] += 1
            stats[s][f'{fonte}_fv'] += int(tem_fv)
        print(f'\r  {i}/{len(tarefas)} arquivos  ({time.time() - t0:.0f}s)', end='', flush=True)
    for ws in writers.values():
        for w in ws:
            w.close()
    print()

    # --- passo 2: embaralhar cada balde e gravar ------------------------------
    print('Passo 2/2: embaralhando baldes e gravando arquivos finais...')
    for s, n in n_bal.items():
        (dir_out / s).mkdir(parents=True, exist_ok=True)
        part = 0
        for k in range(n):
            balde = tmp_dir / f'{s}_{k:05d}.gz'
            regs = ler_registros(tf, balde)
            balde.unlink()
            if not regs:
                continue
            rng.shuffle(regs)
            final = dir_out / s / f'tfrecord_fv_{tag}_{s}_part{part:04d}.tfrecord.gz'
            with tf.io.TFRecordWriter(str(final), options=opts) as w:
                for raw in regs:
                    w.write(raw)
            part += 1
            print(f'\r  {s}: {k + 1}/{n} baldes', end='', flush=True)
        print(f'  → {part} arquivos')
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- resumo / manifesto ---------------------------------------------------
    print('\nRegistros por split:')
    print(f'  {"split":<6} {"v3":>8} {"v3 FV":>8} {"novos":>8} {"novos FV":>9} {"inválidos":>10}')
    for s in SPLITS:
        c = stats[s]
        print(f'  {s:<6} {c["v3"]:>8} {c["v3_fv"]:>8} {c["revisao"]:>8} '
              f'{c["revisao_fv"]:>9} {c["invalido"]:>10}')
    manifesto = {
        'criado_em': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fontes': {'v3': str(dir_v3), 'revisao': str(dir_rev)},
        'parametros': {'tag': tag, 'modo_v3': modo_v3, 'ratios_novos': ratios,
                       'bloco_km': bloco_km, 'registros_por_arquivo': reg_arq, 'seed': seed},
        'arquivos_revisao_sem_coordenadas': sem_coord,
        'registros': {s: dict(c) for s, c in stats.items()},
    }
    (dir_out / 'manifesto.json').write_text(json.dumps(manifesto, indent=2, ensure_ascii=False))
    print(f'\nManifesto → {dir_out / "manifesto.json"}')
    print(f'No notebook: DATASET_ROOT_DS = "<pasta>/{dir_out.name}"')
    print('Concluído.')


if __name__ == '__main__':
    main()
