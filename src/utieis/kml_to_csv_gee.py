#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kml_to_csv_gee.py
=================
Converte um arquivo KML em CSV pronto para upload no Google Earth Engine.

O GEE exige as colunas 'longitude' e 'latitude' para importar CSV como
FeatureCollection de pontos.

Geometrias suportadas:
  - Point      → longitude/latitude diretos
  - Polygon    → centroide do polígono
  - LineString → centroide da linha

Atributos extraídos por Placemark:
  - name
  - description
  - SimpleData (campos de ExtendedData/SchemaData)
  - altitude (quando disponível)

Uso:
  python kml_to_csv_gee.py --input arquivo.kml
  python kml_to_csv_gee.py --input arquivo.kml --output saida.csv
  python kml_to_csv_gee.py --input arquivo.kml --geometry centroid
"""

import csv
import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# Namespaces comuns em arquivos KML
KML_NS  = 'http://www.opengis.net/kml/2.2'
KML_NS2 = 'http://earth.google.com/kml/2.2'
GX_NS   = 'http://www.google.com/kml/ext/2.2'


def detect_namespace(tree_root) -> str:
    """Detecta o namespace KML do arquivo."""
    tag = tree_root.tag
    if tag.startswith('{'):
        return tag[1:tag.index('}')]
    return ''


def parse_coordinates(coords_text: str) -> list:
    """
    Converte texto de coordenadas KML em lista de (lon, lat, alt).
    Aceita tanto ponto único quanto lista de pontos (polígono/linha).
    """
    points = []
    for token in coords_text.strip().split():
        parts = token.split(',')
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                alt = float(parts[2]) if len(parts) > 2 else 0.0
                points.append((lon, lat, alt))
            except ValueError:
                continue
    return points


def centroid(points: list) -> tuple:
    """Retorna o centroide (lon, lat, alt) de uma lista de pontos."""
    if not points:
        return None
    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)
    alt = sum(p[2] for p in points) / len(points)
    return (lon, lat, alt)


def get_text(element, tag: str, ns: str) -> str:
    """Retorna o texto de um sub-elemento ou string vazia."""
    child = element.find(f'{{{ns}}}{tag}')
    if child is not None and child.text:
        return child.text.strip()
    return ''


def extract_placemarks(kml_path: Path, geometry_mode: str) -> list:
    """
    Percorre o KML e extrai todos os Placemarks como lista de dicts.
    geometry_mode: 'point' (apenas Points) | 'centroid' (todos como centroide)
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns   = detect_namespace(root)

    if not ns:
        print(f'Aviso: namespace KML não detectado em {kml_path.name}')
        ns = KML_NS

    # Coleta todos os nomes de campos SimpleData para o cabeçalho
    schema_fields: list = []
    for schema in root.iter(f'{{{ns}}}Schema'):
        for sf in schema.iter(f'{{{ns}}}SimpleField'):
            fname = sf.get('name', '')
            dname_el = sf.find(f'{{{ns}}}displayName')
            dname = dname_el.text.strip() if (dname_el is not None and dname_el.text) else fname
            if fname and fname not in [f['name'] for f in schema_fields]:
                schema_fields.append({'name': fname, 'display': dname})

    rows: list = []

    for pm in root.iter(f'{{{ns}}}Placemark'):
        row = {}

        # Atributos básicos
        row['name']        = get_text(pm, 'name', ns)
        row['description'] = get_text(pm, 'description', ns)
        row['placemark_id'] = pm.get('id', '')

        # SimpleData (campos extras do ExtendedData)
        for sd in pm.iter(f'{{{ns}}}SimpleData'):
            field_name = sd.get('name', '')
            # tenta mapear para displayName
            display = field_name
            for sf in schema_fields:
                if sf['name'] == field_name:
                    display = sf['display']
                    break
            row[display] = sd.text.strip() if sd.text else ''

        # ── Ponto ────────────────────────────────────────────────────────────
        point_el = pm.find(f'{{{ns}}}Point')
        if point_el is not None:
            coords_el = point_el.find(f'{{{ns}}}coordinates')
            if coords_el is not None and coords_el.text:
                pts = parse_coordinates(coords_el.text)
                if pts:
                    row['longitude'] = pts[0][0]
                    row['latitude']  = pts[0][1]
                    row['altitude']  = pts[0][2]
                    row['geometry_type'] = 'Point'
                    rows.append(row)
            continue

        if geometry_mode == 'point':
            # Ignora geometrias que não são pontos
            continue

        # ── Polígono ─────────────────────────────────────────────────────────
        poly_el = pm.find(f'.//{{{ns}}}Polygon')
        if poly_el is not None:
            coords_el = poly_el.find(f'.//{{{ns}}}coordinates')
            if coords_el is not None and coords_el.text:
                pts = parse_coordinates(coords_el.text)
                c   = centroid(pts)
                if c:
                    row['longitude']     = c[0]
                    row['latitude']      = c[1]
                    row['altitude']      = c[2]
                    row['geometry_type'] = 'Polygon_centroid'
                    rows.append(row)
            continue

        # ── LineString ───────────────────────────────────────────────────────
        line_el = pm.find(f'.//{{{ns}}}LineString')
        if line_el is not None:
            coords_el = line_el.find(f'{{{ns}}}coordinates')
            if coords_el is not None and coords_el.text:
                pts = parse_coordinates(coords_el.text)
                c   = centroid(pts)
                if c:
                    row['longitude']     = c[0]
                    row['latitude']      = c[1]
                    row['altitude']      = c[2]
                    row['geometry_type'] = 'LineString_centroid'
                    rows.append(row)

    return rows


def save_csv(rows: list, output_path: Path):
    """Salva lista de dicts como CSV com longitude/latitude como primeiras colunas."""
    if not rows:
        print('Nenhum dado para salvar.')
        return

    # Garante longitude/latitude como primeiras colunas — exigido pelo GEE
    # Coleta todas as chaves únicas de todos os registros
    priority = ['longitude', 'latitude', 'altitude', 'geometry_type', 'name', 'description']
    seen: dict = {}
    for r in rows:
        for k in r:
            seen[k] = True
    all_keys   = list(seen.keys())
    extra      = [k for k in all_keys if k not in priority]
    fieldnames = [k for k in priority if k in seen] + extra

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore',
                                restval='')
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nCSV salvo: {output_path}')
    print(f'Registros: {len(rows)}')
    print(f'Colunas  : {", ".join(fieldnames)}')


def main():
    ap = argparse.ArgumentParser(
        description='Converte KML → CSV com longitude/latitude para upload no GEE')
    ap.add_argument('--input',    required=True, type=Path,
                    help='Arquivo KML de entrada')
    ap.add_argument('--output',   type=Path, default=None,
                    help='Arquivo CSV de saída (padrão: mesmo nome do KML)')
    ap.add_argument('--geometry', choices=['point', 'centroid'], default='centroid',
                    help='Como tratar geometrias: '
                         '"point" ignora não-pontos; '
                         '"centroid" converte polígonos e linhas para centroide (padrão)')
    args = ap.parse_args()

    kml_path = args.input.expanduser().resolve()
    if not kml_path.exists():
        print(f'Erro: arquivo não encontrado: {kml_path}')
        sys.exit(1)

    output_path = args.output or kml_path.with_suffix('.csv')

    print(f'Lendo : {kml_path.name}')
    print(f'Modo  : {args.geometry}')

    rows = extract_placemarks(kml_path, args.geometry)

    if not rows:
        print('Nenhum Placemark encontrado com coordenadas válidas.')
        sys.exit(1)

    # Resumo por tipo de geometria
    types = {}
    for r in rows:
        t = r.get('geometry_type', 'desconhecido')
        types[t] = types.get(t, 0) + 1
    for t, n in types.items():
        print(f'  {t}: {n}')

    save_csv(rows, output_path)


if __name__ == '__main__':
    main()
