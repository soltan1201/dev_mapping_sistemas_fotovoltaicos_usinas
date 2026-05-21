#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador de Tasks TFRecord - Fotovoltaica
=============================================
Lista o status das tasks de exportação TFRecord (tfrecord_fv_*),
cancela as travadas (RUNNING/READY) e salva um relatório.

CONFIGURAÇÕES:
  PREFIX_TASK   → prefixo das tasks a monitorar
  CANCELAR      → True cancela RUNNING/READY; False só lista
  SALVAR_RELATORIO → salva relatorio_tasks_tfrecord.txt no diretório atual
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import collections
collections.Callable = collections.abc.Callable

import ee

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

PREFIX_TASK      = "tfrecord_fv_"   # prefixo das tasks deste pipeline
CANCELAR         = False            # True → cancela RUNNING e READY
SALVAR_RELATORIO = True             # True → grava relatorio_tasks_tfrecord.txt
LIMITE_TASKS     = 3000             # máx tasks a buscar no GEE

# ==============================================================================

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project=projAccount)
    print("Earth Engine inicializado com sucesso!\n")
except Exception as e:
    print("Erro de Inicialização:", e)
    raise


def duracao_str(ms_start, ms_update):
    """Converte timestamps ms → string legível de duração."""
    if ms_start and ms_update:
        d = timedelta(milliseconds=ms_update - ms_start)
        total = int(d.total_seconds())
        h, rem = divmod(total, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return "—"


def get_all_tasks(prefix, limite):
    """Retorna todas as tasks cujo description começa com prefix."""
    raw = ee.data.getTaskList()      # retorna lista de dicts
    return [t for t in raw if t.get("description", "").startswith(prefix)]


# ------------------------------------------------------------------------------
# 1. Buscar tasks
# ------------------------------------------------------------------------------
print(f"Buscando tasks com prefixo '{PREFIX_TASK}'...")
tasks = get_all_tasks(PREFIX_TASK, LIMITE_TASKS)
print(f"Total encontrado: {len(tasks)}\n")

# ------------------------------------------------------------------------------
# 2. Agrupar por estado
# ------------------------------------------------------------------------------
por_estado = {}
for t in tasks:
    estado = t.get("state", "UNKNOWN")
    por_estado.setdefault(estado, []).append(t)

print("=" * 60)
print("RESUMO POR ESTADO")
print("=" * 60)
for estado, lista in sorted(por_estado.items()):
    print(f"  {estado:20s} : {len(lista):5d} tasks")
print("=" * 60)

# ------------------------------------------------------------------------------
# 3. Detalhar tasks problemáticas
# ------------------------------------------------------------------------------
ESTADOS_PROBLEMA = {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}
ESTADOS_RODANDO  = {"RUNNING", "READY", "UNSUBMITTED"}

tasks_problema = por_estado.get("FAILED", []) + por_estado.get("CANCELLED", [])
tasks_rodando  = por_estado.get("RUNNING", []) + por_estado.get("READY", []) + por_estado.get("UNSUBMITTED", [])

if tasks_rodando:
    print(f"\nTASKS EM EXECUÇÃO / FILA ({len(tasks_rodando)}):")
    for t in tasks_rodando:
        dur = duracao_str(t.get("start_timestamp_ms"), t.get("update_timestamp_ms"))
        print(f"  [{t['state']:18s}] {t['description']}  (duração: {dur})")

if tasks_problema:
    print(f"\nTASKS COM PROBLEMA ({len(tasks_problema)}):")
    for t in tasks_problema:
        msg = t.get("error_message", "")
        dur = duracao_str(t.get("start_timestamp_ms"), t.get("update_timestamp_ms"))
        print(f"  [{t['state']:18s}] {t['description']}  {dur}  — {msg[:80]}")

# ------------------------------------------------------------------------------
# 4. Cancelar tasks travadas (se configurado)
# ------------------------------------------------------------------------------
if CANCELAR and tasks_rodando:
    print(f"\nCANCELANDO {len(tasks_rodando)} tasks (RUNNING/READY)...")
    canceladas = 0
    for t in tasks_rodando:
        try:
            ee.data.cancelTask(t["id"])
            print(f"  Cancelada: {t['description']}")
            canceladas += 1
        except Exception as e:
            print(f"  Erro ao cancelar {t['description']}: {e}")
    print(f"  {canceladas}/{len(tasks_rodando)} tasks canceladas.")
elif not CANCELAR and tasks_rodando:
    print(f"\n[INFO] CANCELAR=False → tasks RUNNING/READY mantidas.")
    print("       Mude CANCELAR=True para cancelar.")

# ------------------------------------------------------------------------------
# 5. Salvar relatório
# ------------------------------------------------------------------------------
if SALVAR_RELATORIO:
    relatorio_path = Path("relatorio_tasks_tfrecord.txt")
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(f"Relatório Tasks TFRecord — {agora}\n")
        f.write(f"Prefixo: {PREFIX_TASK}   Total: {len(tasks)}\n\n")

        f.write("RESUMO:\n")
        for estado, lista in sorted(por_estado.items()):
            f.write(f"  {estado:20s}: {len(lista)}\n")
        f.write("\n")

        for estado, lista in sorted(por_estado.items()):
            f.write(f"\n{'=' * 60}\n{estado} ({len(lista)})\n{'=' * 60}\n")
            for t in sorted(lista, key=lambda x: x.get("description", "")):
                dur = duracao_str(t.get("start_timestamp_ms"), t.get("update_timestamp_ms"))
                msg = t.get("error_message", "")
                f.write(f"  {t['description']:<70s}  {dur}  {msg[:80]}\n")

    print(f"\nRelatório salvo em: {relatorio_path.resolve()}")

# ------------------------------------------------------------------------------
# 6. Nomes das tasks que precisam ser resubmetidas
# ------------------------------------------------------------------------------
nomes_para_resubmeter = [t["description"] for t in tasks_problema]
if nomes_para_resubmeter:
    print(f"\nTASKS PARA RESUBMETER ({len(nomes_para_resubmeter)}):")
    for nome in sorted(nomes_para_resubmeter):
        print(f"  {nome}")
    pendentes_path = Path("tasks_para_resubmeter.txt")
    with open(pendentes_path, "w") as f:
        f.write("\n".join(sorted(nomes_para_resubmeter)) + "\n")
    print(f"\nLista salva em: {pendentes_path.resolve()}")
    print("Execute o script principal com SKIP_COMPLETED=True para reprocessar só as pendentes.")
else:
    print("\nNenhuma task para resubmeter.")

print("\nConcluído.")
