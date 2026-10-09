"""Filter the full interaction dataset to include only selected items.

The ratings file is processed in smaller chunks to avoid using too much memory.
Each interaction is linked to its product domain and saved as a Parquet file.

The remaining data-processing steps use the Parquet file because it is faster
to load and process than the original CSV file.
"""
from __future__ import annotations

# Allow this module to import the shared modules at the top of src/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import load_config

cfg = load_config()
RAW = Path(cfg["paths"]["raw_data"]) / cfg["sources"]["interactions_file"]
PROCESSED = Path(cfg["paths"]["processed_data"])
PROCESSED.mkdir(parents=True, exist_ok=True)

IN_ITEMS = PROCESSED / "items_domained.parquet"
OUT = PROCESSED / "interactions_inscope.parquet"

CHUNK = 2_000_000

SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("parent_asin", pa.string()),
    ("rating", pa.float32()),
    ("timestamp", pa.int64()),
    ("domain", pa.string()),
])


def main() -> None:
    items = pd.read_parquet(IN_ITEMS, columns=["parent_asin", "domain"])
    domain_of = dict(zip(items["parent_asin"], items["domain"]))
    print(f"in-scope items : {len(domain_of):,}")
    print(f"reading        : {RAW}")
    print(f"writing        : {OUT}\n")

    writer = None
    n_in = n_out = 0

    reader = pd.read_csv(
        RAW,
        chunksize=CHUNK,
        dtype={"user_id": "string", "parent_asin": "string",
               "rating": "float32", "timestamp": "int64"},
    )

    for chunk in tqdm(reader, unit=" chunks"):
        n_in += len(chunk)
        chunk["domain"] = chunk["parent_asin"].map(domain_of)
        keep = chunk[chunk["domain"].notna()]
        if keep.empty:
            continue
        table = pa.Table.from_pandas(
            keep[["user_id", "parent_asin", "rating", "timestamp", "domain"]],
            schema=SCHEMA, preserve_index=False,
        )
        if writer is None:
            writer = pq.ParquetWriter(OUT, SCHEMA, compression="snappy")
        writer.write_table(table)
        n_out += len(keep)

    if writer is not None:
        writer.close()

    print(f"\ninteractions read : {n_in:,}")
    print(f"interactions kept : {n_out:,}  ({n_out / n_in:.1%})")
    print(f"parquet size      : {OUT.stat().st_size / 1024**2:,.0f} MB")


if __name__ == "__main__":
    main()