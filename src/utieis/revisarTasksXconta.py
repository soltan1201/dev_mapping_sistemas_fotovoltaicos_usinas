#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
#SCRIPT DE CLASSIFICACAO POR BACIA
#Produzido por Geodatin - Dados e Geoinformacao
#DISTRIBUIDO COM GPLv2
'''
import os
import ee 

import sys
import collections
collections.Callable = collections.abc.Callable

from pathlib import Path
pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account, get_project_from_account
from gee_tools import *
projAccount = get_current_account()
print(f"projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize( project= projAccount)
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

# sys.setrecursionlimit(1000000000)

contas = {
    '0': 'caatinga01',
    '1': 'caatinga02',
    '2': 'caatinga03',
    '3': 'caatinga04',
    '4': 'caatinga05',
    '5': 'solkan1201',
    '6': 'solkanGeodatin',
    '8': 'superconta',
    '9': 'solkanCengine',
}

unicaconta = input("É única conta (Y/N): ").strip().upper() == 'Y'
conta_escolhida = int(input("Qual conta deseja usar: ").strip()) if unicaconta else None
numero_tasks = int(input("Quantas run tasks deseja visualizar: ").strip())
cancelar = input("Vai eliminar as tasks da nova conta (Y/N): ").strip().upper() == 'Y'

numeroLimit = max(int(k) for k in contas.keys())

def gerenciador(cont):
    if str(cont) in contas:
        switch_user(contas[str(cont)])
        projAccount = get_project_from_account(contas[str(cont)])
        try:
            ee.Initialize(project=projAccount)
        except ee.EEException:
            print('The Earth Engine package failed to initialize!')
            return cont + 1

        print(f"\n--- Conta: {contas[str(cont)]} ---")
        tarefas = tasks(n=numero_tasks, return_list=True)
        for lin in tarefas:
            print(lin)

    elif cont > numeroLimit:
        return 0
    return cont + 1

if unicaconta:
    print(f"Mudando para conta #{conta_escolhida} <> {contas[str(conta_escolhida)]}")
    gerenciador(conta_escolhida)
    if cancelar:
        cancel(opentasks=True)
else:
    for ii in range(0, numeroLimit + 1):
        gerenciador(ii)
        if cancelar:
            cancel(opentasks=True)
