"""Out-of-core preprocessing for the 24M-row IBM transaction CSV.

DuckDB performs the external chronological sort and may spill to disk.  Only a
bounded pandas chunk is materialized while compact NumPy memmaps are written.
All fitted feature statistics and all sampling statistics use the train prefix.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch

from .graph import TransactionGraph


CACHE_SCHEMA = 1
FEATURE_NAMES = [
    "amount_minmax", "amount_missing", "refund",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "weekend",
    "use_chip_code", "mcc_code", "has_error",
    "merchant_location_frequency", "merchant_location_oov",
]


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on optional backend
        raise RuntimeError(
            "Backend out-of-core memerlukan duckdb. Jalankan `pip install duckdb`."
        ) from exc
    return duckdb


def _literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def _normalized(column):
    return f"coalesce(nullif(trim(cast({column} as varchar)), ''), '<missing>')"


def _parsed_query(csv_path, max_rows):
    scan = (
        f"read_csv({_literal(Path(csv_path).resolve().as_posix())}, header=true, "
        "auto_detect=true, all_varchar=true, sample_size=20480)"
    )
    source = f"select *, row_number() over () - 1 as source_row from {scan}"
    if max_rows is not None:
        source = f"select * from ({source}) source_limit limit {int(max_rows)}"
    timestamp = (
        "try_strptime(concat(trim(\"Year\"), '-', lpad(trim(\"Month\"), 2, '0'), '-', "
        "lpad(trim(\"Day\"), 2, '0'), ' ', trim(\"Time\")), "
        "['%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S'])"
    )
    amount = (
        "try_cast(replace(replace(trim(cast(\"Amount\" as varchar)), '$', ''), ',', '') "
        "as double)"
    )
    return f"""
        with source as ({source})
        select
            cast(source_row as bigint) as source_row,
            cast("User" as varchar) as user_key,
            cast("Merchant Name" as varchar) as merchant_key,
            {_normalized('"Merchant City"')} as city_key,
            {_normalized('"Merchant State"')} as state_key,
            {_normalized('"Zip"')} as zip_key,
            {_normalized('"Use Chip"')} as use_chip_key,
            {_normalized('"MCC"')} as mcc_key,
            {_normalized('"Errors?"')} as errors_key,
            {timestamp} as ts,
            {amount} as amount,
            case lower(trim(cast("Is Fraud?" as varchar)))
                when 'yes' then 1 when 'no' then 0 else null end as label
        from source
    """


def _configure(connection, cache_dir, cfg):
    temp = Path(cache_dir) / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    memory = cfg.get("duckdb_memory_limit", "8GB")
    threads = int(cfg.get("cpu_threads", 4))
    connection.execute(f"set memory_limit={_literal(memory)}")
    connection.execute(f"set threads={threads}")
    connection.execute(f"set temp_directory={_literal(temp.resolve().as_posix())}")
    connection.execute("set preserve_insertion_order=false")


def _sort_csv(connection, csv_path, parquet_path, max_rows, chunk_rows):
    parsed = _parsed_query(csv_path, max_rows)
    audit = connection.execute(f"""
        select count(*) as rows,
               count(*) filter (where ts is null) as invalid_time,
               count(*) filter (where label is null) as invalid_label,
               count(*) filter (where user_key is null or merchant_key is null) as invalid_entity
        from ({parsed}) parsed
    """).fetchone()
    if audit[0] < 3:
        raise ValueError("Dataset memerlukan setidaknya tiga baris")
    if any(int(value) for value in audit[1:]):
        raise ValueError(
            "Data tidak valid: "
            f"timestamp={audit[1]}, label={audit[2]}, entity={audit[3]}"
        )
    parquet_path.unlink(missing_ok=True)
    ordered = f"""
        with parsed as ({parsed})
        select row_number() over (order by ts, source_row) - 1 as tx_id, *
        from parsed
        order by ts, source_row
    """
    connection.execute(
        f"copy ({ordered}) to {_literal(parquet_path.resolve().as_posix())} "
        f"(format parquet, compression zstd, row_group_size {int(chunk_rows)})"
    )
    return int(audit[0])


def _tie_safe_boundaries(connection, parquet_path, total, ratios):
    relation = f"read_parquet({_literal(parquet_path.resolve().as_posix())})"
    nominal = [int(total * ratios[0]), int(total * (ratios[0] + ratios[1]))]
    bounds = []
    for position in nominal:
        timestamp = connection.execute(
            f"select ts from {relation} where tx_id={position}"
        ).fetchone()[0]
        # Keep an equal-timestamp group wholly on the earlier side.  The ratio
        # can move by one timestamp group but chronological information cannot.
        bound = connection.execute(
            f"select count(*) from {relation} where ts <= ?", [timestamp]
        ).fetchone()[0]
        bounds.append(int(bound))
    train_end, val_end = bounds
    if not 0 < train_end < val_end < total:
        raise ValueError("Split temporal tie-safe menghasilkan subset kosong")
    return train_end, val_end


def _feature_state(connection, parquet_path, train_end):
    relation = f"read_parquet({_literal(parquet_path.resolve().as_posix())})"
    median, minimum, maximum = connection.execute(f"""
        select median(amount), min(amount), max(amount)
        from {relation} where tx_id < {train_end}
    """).fetchone()
    median = float(median) if median is not None else 0.0
    minimum = float(minimum) if minimum is not None else median
    maximum = float(maximum) if maximum is not None else median
    categories = {}
    for column in ("use_chip_key", "mcc_key"):
        rows = connection.execute(f"""
            select distinct {column} from {relation}
            where tx_id < {train_end} order by {column}
        """).fetchall()
        categories[column] = [str(row[0]) for row in rows]
    locations = connection.execute(f"""
        select concat(city_key, '|', state_key, '|', zip_key) as location,
               count(*)::double / {train_end} as frequency
        from {relation} where tx_id < {train_end}
        group by location
    """).fetchall()
    return {
        "version": 1,
        "fit_scope": "train_only",
        "amount": {"median": median, "min": minimum, "max": maximum},
        "categories": categories,
        "location_frequency": {str(key): float(value) for key, value in locations},
    }


def _global_codes(values, mapping):
    local, uniques = pd.factorize(values, sort=False)
    if (local < 0).any():
        raise ValueError("Identifier User/Merchant tidak boleh kosong")
    translated = np.empty(len(uniques), dtype=np.int64)
    for index, value in enumerate(uniques):
        key = str(value)
        code = mapping.get(key)
        if code is None:
            code = len(mapping)
            mapping[key] = code
        translated[index] = code
    return translated[local]


def _category_codes(values, categories):
    mapping = {value: index + 1 for index, value in enumerate(categories)}
    return values.map(mapping).fillna(0).to_numpy(np.int64)


def _transform(chunk, state):
    count = len(chunk)
    features = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
    amount_stats = state["amount"]
    amount_series = pd.to_numeric(chunk["amount"], errors="coerce")
    raw = amount_series.fillna(amount_stats["median"]).to_numpy(np.float64)
    scale = amount_stats["max"] - amount_stats["min"] or 1.0
    scaled = (raw - amount_stats["min"]) / scale
    ts = pd.to_datetime(chunk["ts"], errors="raise")
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    dow = ts.dt.dayofweek.to_numpy()
    month = ts.dt.month.to_numpy() - 1
    use_codes = _category_codes(chunk["use_chip_key"], state["categories"]["use_chip_key"])
    mcc_codes = _category_codes(chunk["mcc_key"], state["categories"]["mcc_key"])
    location = (
        chunk["city_key"].astype(str) + "|" + chunk["state_key"].astype(str)
        + "|" + chunk["zip_key"].astype(str)
    )
    location_frequency = location.map(state["location_frequency"])
    columns = [
        np.clip(scaled, 0.0, 1.0), amount_series.isna().to_numpy(), raw < 0,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        (dow >= 5),
        use_codes / max(len(state["categories"]["use_chip_key"]), 1),
        mcc_codes / max(len(state["categories"]["mcc_key"]), 1),
        chunk["errors_key"].ne("<missing>").to_numpy(),
        location_frequency.fillna(0).to_numpy(),
        location_frequency.isna().to_numpy(),
    ]
    features[:] = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("Fitur non-finite ditemukan setelah transformasi chunk")
    diagnostics = {
        "amount_outside_train_range": (scaled < 0) | (scaled > 1),
        "use_chip_oov": use_codes == 0,
        "mcc_oov": mcc_codes == 0,
        "location_oov": location_frequency.isna().to_numpy(),
    }
    return features, use_codes.astype(np.int16), diagnostics


def _allocate(path, dtype, shape):
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def _fill_arrays(connection, parquet_path, cache_dir, total, train_end, val_end,
                 chunk_rows, state, storage_dtype):
    features = _allocate(cache_dir / "features.bin", storage_dtype, (total, len(FEATURE_NAMES)))
    labels = _allocate(cache_dir / "labels.bin", "uint8", (total,))
    entities = _allocate(cache_dir / "entities.bin", "int64", (total, 2))
    timestamps = _allocate(cache_dir / "timestamps.bin", "int64", (total,))
    source_rows = _allocate(cache_dir / "source_rows.bin", "int64", (total,))
    channels = _allocate(cache_dir / "channels.bin", "int16", (total,))
    user_map, merchant_map = {}, {}
    names = ("amount_outside_train_range", "use_chip_oov", "mcc_oov", "location_oov")
    diagnostics = {split: {key: 0 for key in names} for split in ("train", "val", "test")}
    split_ranges = (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, total))
    relation = f"read_parquet({_literal(parquet_path.resolve().as_posix())})"
    cursor = connection.execute(f"select * from {relation} order by tx_id")
    vectors = max(1, math.ceil(chunk_rows / 2048))
    offset = 0
    while True:
        chunk = cursor.fetch_df_chunk(vectors)
        if chunk is None or chunk.empty:
            break
        end = offset + len(chunk)
        if int(chunk["tx_id"].iloc[0]) != offset:
            raise RuntimeError("Urutan hasil DuckDB tidak kontinu")
        values, channel_codes, report = _transform(chunk, state)
        features[offset:end] = values.astype(storage_dtype, copy=False)
        labels[offset:end] = chunk["label"].to_numpy(np.uint8)
        entities[offset:end, 0] = _global_codes(chunk["user_key"], user_map)
        entities[offset:end, 1] = _global_codes(chunk["merchant_key"], merchant_map)
        timestamps[offset:end] = pd.to_datetime(chunk["ts"]).to_numpy("datetime64[ns]").view(np.int64)
        source_rows[offset:end] = chunk["source_row"].to_numpy(np.int64)
        channels[offset:end] = channel_codes
        for split, lo, hi in split_ranges:
            local_lo, local_hi = max(lo, offset) - offset, min(hi, end) - offset
            if local_hi > local_lo:
                for key, flags in report.items():
                    diagnostics[split][key] += int(np.asarray(flags[local_lo:local_hi]).sum())
        offset = end
    if offset != total:
        raise RuntimeError(f"DuckDB menghasilkan {offset} baris; diharapkan {total}")
    num_users, num_merchants = len(user_map), len(merchant_map)
    for start in range(0, total, chunk_rows):
        entities[start:start + chunk_rows, 1] += num_users
    for array in (features, labels, entities, timestamps, source_rows, channels):
        array.flush()
    return num_users, num_merchants, diagnostics


def _fill_csr(cache_dir, total, train_end, num_entities, num_users, chunk_rows):
    entities = np.memmap(cache_dir / "entities.bin", dtype="int64", mode="r+", shape=(total, 2))
    labels = np.memmap(cache_dir / "labels.bin", dtype="uint8", mode="r+", shape=(total,))
    degree = np.zeros(num_entities, dtype=np.int64)
    fraud_count = np.zeros(num_entities, dtype=np.int64)
    for start in range(0, train_end, chunk_rows):
        end = min(start + chunk_rows, train_end)
        endpoint = entities[start:end].reshape(-1)
        degree += np.bincount(endpoint, minlength=num_entities)
        fraud_count += np.bincount(
            endpoint, weights=np.repeat(labels[start:end], 2), minlength=num_entities
        ).astype(np.int64)
    rowptr = np.empty(num_entities + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(degree, out=rowptr[1:])
    col = _allocate(cache_dir / "col.bin", "int64", (2 * train_end,))
    cursors = rowptr[:-1].copy()
    for start in range(0, train_end, chunk_rows):
        end = min(start + chunk_rows, train_end)
        ids = entities[start:end].T.reshape(-1)
        transactions = np.tile(np.arange(start, end, dtype=np.int64), 2)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        group_start = np.empty(len(order), dtype=bool)
        group_start[0] = True
        group_start[1:] = sorted_ids[1:] != sorted_ids[:-1]
        starts = np.maximum.accumulate(np.where(group_start, np.arange(len(order)), 0))
        rank = np.arange(len(order)) - starts
        destinations = cursors[sorted_ids] + rank
        col[destinations] = transactions[order]
        cursors += np.bincount(sorted_ids, minlength=num_entities)
    if not np.array_equal(cursors, rowptr[1:]):
        raise RuntimeError("CSR adjacency tidak terisi tepat")
    col.flush()
    for name, values in (("rowptr.bin", rowptr), ("degree.bin", degree), ("fraud_count.bin", fraud_count)):
        target = _allocate(cache_dir / name, "int64", values.shape)
        target[:] = values
        target.flush()
    merchant_local = entities[:train_end, 1] - num_users
    keys = entities[:train_end, 0] * max(num_entities - num_users, 1) + merchant_local
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    pair_count = _allocate(cache_dir / "pair_count.bin", "int32", (train_end,))
    pair_count[:] = counts[inverse].astype(np.int32, copy=False)
    pair_count.flush()
    return degree


def _split_stats(labels, timestamps, ranges):
    output = {}
    for name, lo, hi in ranges:
        positives = int(np.asarray(labels[lo:hi], dtype=np.int64).sum())
        if not 0 < positives < hi - lo:
            raise ValueError(
                f"Split {name} memerlukan normal dan fraud; ditemukan {positives}/{hi-lo}"
            )
        output[name] = {
            "total": hi - lo,
            "fraud": positives,
            "fraud_rate": positives / (hi - lo),
            "start": pd.Timestamp(int(timestamps[lo]), unit="ns").isoformat(),
            "end": pd.Timestamp(int(timestamps[hi - 1]), unit="ns").isoformat(),
        }
    return output


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _open(cache_dir, manifest):
    def tensor(name, dtype, shape):
        array = np.memmap(cache_dir / name, dtype=dtype, mode="r+", shape=tuple(shape))
        return torch.from_numpy(array)

    shapes = manifest["shapes"]
    graph = TransactionGraph(
        tensor("features.bin", manifest["storage_dtype"], shapes["features"]),
        tensor("labels.bin", "uint8", shapes["labels"]),
        tensor("entities.bin", "int64", shapes["entities"]),
        tensor("rowptr.bin", "int64", shapes["rowptr"]),
        tensor("col.bin", "int64", shapes["col"]),
        tensor("degree.bin", "int64", shapes["degree"]),
        tensor("fraud_count.bin", "int64", shapes["fraud_count"]),
        tensor("pair_count.bin", "int32", shapes["pair_count"]),
        tensor("source_rows.bin", "int64", shapes["source_rows"]),
        tensor("timestamps.bin", "int64", shapes["timestamps"]),
        int(manifest["train_end"]), int(manifest["val_end"]),
        int(manifest["num_users"]), manifest["metadata"],
        tensor("channels.bin", "int16", shapes["channels"]),
    )
    return graph


def prepare_graph_out_of_core(csv_path, max_rows, ratios, feature_config, runtime,
                              cache_root, fingerprint, rebuild=False):
    """Return a disk-backed ``TransactionGraph`` compatible with EXP12."""
    if feature_config.get("encoder") != "proposal":
        raise ValueError("Backend duckdb saat ini khusus features.encoder=proposal")
    cache_dir = Path(cache_root) / f"stream_{fingerprint[:16]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != CACHE_SCHEMA or manifest.get("fingerprint") != fingerprint:
            raise ValueError("Manifest cache streaming tidak cocok")
        return _open(cache_dir, manifest), manifest_path

    duckdb = _duckdb()
    connection = duckdb.connect()
    _configure(connection, cache_dir, runtime)
    chunk_rows = int(runtime.get("preprocess_chunk_rows", 250_000))
    parquet_path = cache_dir / "chronological.parquet"
    total = _sort_csv(connection, csv_path, parquet_path, max_rows, chunk_rows)
    train_end, val_end = _tie_safe_boundaries(connection, parquet_path, total, ratios)
    state = _feature_state(connection, parquet_path, train_end)
    num_users, num_merchants, diagnostics = _fill_arrays(
        connection, parquet_path, cache_dir, total, train_end, val_end, chunk_rows, state,
        feature_config.get("storage_dtype", "float16"),
    )
    connection.close()
    degree = _fill_csr(
        cache_dir, total, train_end, num_users + num_merchants, num_users, chunk_rows
    )
    labels = np.memmap(cache_dir / "labels.bin", dtype="uint8", mode="r+", shape=(total,))
    timestamps = np.memmap(cache_dir / "timestamps.bin", dtype="int64", mode="r+", shape=(total,))
    entities = np.memmap(cache_dir / "entities.bin", dtype="int64", mode="r+", shape=(total, 2))
    ranges = (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, total))
    split_stats = _split_stats(labels, timestamps, ranges)
    drift = {}
    for name, lo, hi in ranges:
        drift[name] = {
            "amount_outside_train_range": diagnostics[name]["amount_outside_train_range"] / (hi - lo),
            "use_chip_oov_rate": diagnostics[name]["use_chip_oov"] / (hi - lo),
            "mcc_oov_rate": diagnostics[name]["mcc_oov"] / (hi - lo),
            "location_oov_rate": diagnostics[name]["location_oov"] / (hi - lo),
        }
        if name != "train":
            drift[name]["unseen_user_rate"] = float((degree[entities[lo:hi, 0]] == 0).mean())
            drift[name]["unseen_merchant_rate"] = float((degree[entities[lo:hi, 1]] == 0).mean())
    channel_names = ["<oov>"] + state["categories"]["use_chip_key"]
    metadata = {
        "feature_names": FEATURE_NAMES,
        "encoder": state,
        "split_stats": split_stats,
        "drift": drift,
        "context_policy": "frozen_train_history",
        "feature_config": feature_config,
        "feature_storage_bytes": total * len(FEATURE_NAMES) * np.dtype(feature_config["storage_dtype"]).itemsize,
        "payment_channel_names": channel_names,
        "history_nodes": train_end + int((degree > 0).sum()),
        "history_directed_edges": 4 * train_end,
        "tie_at_split_boundary": {"train_val": False, "val_test": False},
        "preprocess_backend": "duckdb_memmap",
        "preprocess_chunk_rows": chunk_rows,
        "sampling_statistics_scope": "train_only",
    }
    shapes = {
        "features": [total, len(FEATURE_NAMES)], "labels": [total],
        "entities": [total, 2], "rowptr": [num_users + num_merchants + 1],
        "col": [2 * train_end], "degree": [num_users + num_merchants],
        "fraud_count": [num_users + num_merchants], "pair_count": [train_end],
        "source_rows": [total], "timestamps": [total], "channels": [total],
    }
    manifest = {
        "schema": CACHE_SCHEMA, "fingerprint": fingerprint,
        "storage_dtype": feature_config["storage_dtype"], "shapes": shapes,
        "train_end": train_end, "val_end": val_end, "num_users": num_users,
        "num_merchants": num_merchants, "metadata": metadata,
    }
    _write_json(manifest_path, manifest)
    parquet_path.unlink(missing_ok=True)
    return _open(cache_dir, manifest), manifest_path
