#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Подготовка большого файла all nodes.csv к формату "один синхронизированный файл на узел".

Что делает скрипт:
1) Читает большой CSV чанками, чтобы не загружать всё в память.
2) Оставляет только метрики CPU и +12V.
3) Разбивает строки по узлам во временные raw-файлы.
4) Для каждого узла:
   - сортирует точки по времени;
   - строит единую временную сетку;
   - CPU приводит к сетке через resample + interpolation;
   - voltage приводит к сетке через resample + ffill/bfill;
   - считает cpu_z, voltage_z, relation_signal;
   - сохраняет итоговый файл в data/prepared_nodes/<node>.csv
5) Сохраняет summary CSV по всем узлам.

Пример запуска из корня проекта:
python .\data_analysis_module\prepare_all_nodes.py ^
  --input_csv .\data\all nodes.csv ^
  --output_dir .\data\prepared_nodes ^
  --resample_rule 30s
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


CPU_KEY_CANDIDATES = {
    "system.cpu.util",
}

VOLTAGE_KEY_PATTERNS = (
    "+12v",
    "ipmi.value[+12v]",
)

CPU_ELEMENT_PATTERNS = (
    "cpu utilization",
)

VOLTAGE_ELEMENT_PATTERNS = (
    "+12v",
    "12v",
)

DEFAULT_CHUNK_SIZE = 200_000


def norm_text(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-\.]+", "_", str(name).strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    return cleaned or "unknown_node"


def robust_zscore(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    median = float(values.median())
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        std = float(values.std(ddof=0))
        scale = std if std > eps else 1.0
    return (values - median) / scale


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = [str(c).strip() for c in columns]
    lowered = {c.lower(): c for c in cols}
    compact = {re.sub(r"[\s_]+", "", c.lower()): c for c in cols}

    for cand in candidates:
        c1 = cand.lower().strip()
        if c1 in lowered:
            return lowered[c1]

    for cand in candidates:
        c2 = re.sub(r"[\s_]+", "", cand.lower().strip())
        if c2 in compact:
            return compact[c2]

    return None


@dataclass
class ColumnMap:
    host: str
    itemid: Optional[str]
    element: Optional[str]
    key: Optional[str]
    value: str
    ts: str


def detect_columns(df_columns: Iterable[str]) -> ColumnMap:
    host = find_column(df_columns, ["Хост", "host", "hostname"])
    itemid = find_column(df_columns, ["itemid"])
    element = find_column(df_columns, ["Элемент", "element", "item", "name"])
    key = find_column(df_columns, ["Ключ", "key"])
    value = find_column(df_columns, ["Числовое значение", "numeric value", "value"])
    ts = find_column(df_columns, ["Время числового значения", "timestamp", "time", "clock", "unix timestamp"])

    missing = []
    if not host:
        missing.append("Хост / host")
    if not value:
        missing.append("Числовое значение / value")
    if not ts:
        missing.append("Время числового значения / timestamp")

    if missing:
        raise KeyError(
            "Не удалось определить обязательные колонки: "
            + ", ".join(missing)
            + f". Доступные колонки: {list(df_columns)}"
        )

    return ColumnMap(
        host=host,
        itemid=itemid,
        element=element,
        key=key,
        value=value,
        ts=ts,
    )


def is_cpu_row(row: pd.Series, cmap: ColumnMap) -> bool:
    key_val = norm_text(row[cmap.key]) if cmap.key else ""
    elem_val = norm_text(row[cmap.element]) if cmap.element else ""

    if key_val in CPU_KEY_CANDIDATES:
        return True
    if any(p in elem_val for p in CPU_ELEMENT_PATTERNS):
        return True
    return False


def is_voltage_row(row: pd.Series, cmap: ColumnMap) -> bool:
    key_val = norm_text(row[cmap.key]) if cmap.key else ""
    elem_val = norm_text(row[cmap.element]) if cmap.element else ""

    if any(p in key_val for p in VOLTAGE_KEY_PATTERNS):
        return True
    if any(p in elem_val for p in VOLTAGE_ELEMENT_PATTERNS):
        return True
    return False


def split_big_file_to_node_raw(
    input_csv: Path,
    tmp_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: Optional[str] = None,
) -> Dict[str, Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    node_to_file: Dict[str, Path] = {}
    written_headers = set()
    cmap: Optional[ColumnMap] = None

    read_kwargs = {
        "chunksize": chunk_size,
        "low_memory": False,
    }
    if encoding:
        read_kwargs["encoding"] = encoding

    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, **read_kwargs), start=1):
        if cmap is None:
            cmap = detect_columns(chunk.columns)

        use_cols = [cmap.host, cmap.value, cmap.ts]
        if cmap.itemid:
            use_cols.append(cmap.itemid)
        if cmap.element:
            use_cols.append(cmap.element)
        if cmap.key:
            use_cols.append(cmap.key)

        part = chunk[use_cols].copy()

        part[cmap.value] = pd.to_numeric(part[cmap.value], errors="coerce")
        part[cmap.ts] = pd.to_numeric(part[cmap.ts], errors="coerce")
        part = part.dropna(subset=[cmap.host, cmap.value, cmap.ts])

        if part.empty:
            continue

        metric_type = []
        for _, row in part.iterrows():
            if is_cpu_row(row, cmap):
                metric_type.append("cpu")
            elif is_voltage_row(row, cmap):
                metric_type.append("voltage")
            else:
                metric_type.append(None)

        part["metric_type"] = metric_type
        part = part.dropna(subset=["metric_type"])
        if part.empty:
            continue

        for host_name, host_df in part.groupby(cmap.host, sort=False):
            host_name = str(host_name).strip()
            safe_host = sanitize_filename(host_name)
            node_path = tmp_dir / f"{safe_host}.csv"
            node_to_file[safe_host] = node_path

            header = not node_path.exists() or safe_host not in written_headers
            host_df = host_df[[cmap.host, "metric_type", cmap.ts, cmap.value]].copy()
            host_df.columns = ["host", "metric_type", "timestamp_unix", "value"]
            host_df.to_csv(
                node_path,
                mode="a",
                header=header,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            written_headers.add(safe_host)

        print(f"[split] chunk={chunk_idx} processed")

    return node_to_file


def build_aligned_node_file(
    raw_node_csv: Path,
    output_csv: Path,
    resample_rule: str = "30s",
    min_points: int = 20,
) -> Optional[dict]:
    df = pd.read_csv(raw_node_csv, low_memory=False)
    if df.empty:
        return None

    df["timestamp_unix"] = pd.to_numeric(df["timestamp_unix"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp_unix", "value"])

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp_unix"].astype("int64"), unit="s", utc=True).dt.tz_convert(None)

    cpu = (
        df[df["metric_type"] == "cpu"][["timestamp", "value"]]
        .rename(columns={"value": "cpu"})
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )

    voltage = (
        df[df["metric_type"] == "voltage"][["timestamp", "value"]]
        .rename(columns={"value": "voltage"})
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )

    if cpu.empty or voltage.empty:
        return None

    start_ts = max(cpu["timestamp"].min(), voltage["timestamp"].min())
    end_ts = min(cpu["timestamp"].max(), voltage["timestamp"].max())

    if start_ts >= end_ts:
        return None

    cpu = cpu[(cpu["timestamp"] >= start_ts) & (cpu["timestamp"] <= end_ts)].copy()
    voltage = voltage[(voltage["timestamp"] >= start_ts) & (voltage["timestamp"] <= end_ts)].copy()

    if cpu.empty or voltage.empty:
        return None

    cpu_grid = (
        cpu.set_index("timestamp")
        .resample(resample_rule)
        .mean()
        .interpolate(method="time", limit_direction="both")
        .reset_index()
    )

    voltage_grid = (
        voltage.set_index("timestamp")
        .resample(resample_rule)
        .last()
        .ffill()
        .bfill()
        .reset_index()
    )

    start_grid = max(cpu_grid["timestamp"].min(), voltage_grid["timestamp"].min())
    end_grid = min(cpu_grid["timestamp"].max(), voltage_grid["timestamp"].max())

    if start_grid >= end_grid:
        return None

    grid = pd.DataFrame({"timestamp": pd.date_range(start_grid, end_grid, freq=resample_rule)})

    aligned = (
        grid.merge(cpu_grid, on="timestamp", how="left")
        .merge(voltage_grid, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    aligned["cpu"] = aligned["cpu"].interpolate(method="linear", limit_direction="both")
    aligned["voltage"] = aligned["voltage"].ffill().bfill()
    aligned = aligned.dropna(subset=["cpu", "voltage"]).reset_index(drop=True)

    if len(aligned) < min_points:
        return None

    aligned["cpu_z"] = robust_zscore(aligned["cpu"])
    aligned["voltage_z"] = robust_zscore(aligned["voltage"])
    aligned["relation_signal"] = aligned["cpu_z"] + aligned["voltage_z"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_csv, index=False)

    return {
        "node": raw_node_csv.stem,
        "rows": int(len(aligned)),
        "start_time": aligned["timestamp"].iloc[0],
        "end_time": aligned["timestamp"].iloc[-1],
        "cpu_mean": float(aligned["cpu"].mean()),
        "voltage_mean": float(aligned["voltage"].mean()),
        "output_file": str(output_csv),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовка all nodes.csv в набор синхронизированных processed-файлов по узлам"
    )
    parser.add_argument("--input_csv", type=str, default=None, help="Путь к data/all nodes.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Папка назначения, например data/prepared_nodes")
    parser.add_argument("--temp_dir", type=str, default=None, help="Временная папка для raw-файлов по узлам")
    parser.add_argument("--resample_rule", type=str, default="30s", help="Шаг временной сетки, например 30s")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Размер чанка при чтении CSV")
    parser.add_argument("--encoding", type=str, default=None, help="Необязательно: кодировка CSV")
    parser.add_argument("--keep_temp", action="store_true", help="Не удалять временные raw-файлы")
    return parser


def resolve_default_paths(args) -> tuple[Path, Path, Path]:
    cwd = Path.cwd()
    default_input = cwd / "data" / "all nodes.csv"
    default_output = cwd / "data" / "prepared_nodes"
    default_temp = default_output / "_tmp_raw"

    input_csv = Path(args.input_csv) if args.input_csv else default_input
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    temp_dir = Path(args.temp_dir) if args.temp_dir else default_temp

    return input_csv.resolve(), output_dir.resolve(), temp_dir.resolve()


def main() -> None:
    args = build_parser().parse_args()
    input_csv, output_dir, temp_dir = resolve_default_paths(args)

    if not input_csv.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] input={input_csv}")
    print(f"[start] output_dir={output_dir}")
    print(f"[start] temp_dir={temp_dir}")
    print("[stage 1] split big file by node")

    node_to_raw = split_big_file_to_node_raw(
        input_csv=input_csv,
        tmp_dir=temp_dir,
        chunk_size=args.chunk_size,
        encoding=args.encoding,
    )

    print(f"[stage 1] nodes found={len(node_to_raw)}")
    print("[stage 2] build aligned files")

    summary_rows = []
    skipped_nodes = []

    for idx, (node_name, raw_path) in enumerate(sorted(node_to_raw.items()), start=1):
        out_path = output_dir / f"{node_name}.csv"
        try:
            meta = build_aligned_node_file(
                raw_node_csv=raw_path,
                output_csv=out_path,
                resample_rule=args.resample_rule,
            )
            if meta is None:
                skipped_nodes.append(node_name)
                print(f"[align] {idx}/{len(node_to_raw)} node={node_name} skipped")
            else:
                summary_rows.append(meta)
                print(f"[align] {idx}/{len(node_to_raw)} node={node_name} rows={meta['rows']}")
        except Exception as e:
            skipped_nodes.append(node_name)
            print(f"[align] {idx}/{len(node_to_raw)} node={node_name} failed: {e}")

    summary_path = output_dir / "_prepared_nodes_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    skipped_path = output_dir / "_skipped_nodes.txt"
    with open(skipped_path, "w", encoding="utf-8") as f:
        for node in skipped_nodes:
            f.write(str(node) + "\n")

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[done]")
    print(f"prepared files dir: {output_dir}")
    print(f"summary: {summary_path}")
    print(f"skipped nodes list: {skipped_path}")


if __name__ == "__main__":
    main()
