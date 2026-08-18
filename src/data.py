from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "Cellulose(wt%)": "cellulose_pct",
    "Hemicellulose(wt%)": "hemicellulose_pct",
    "Lignin(wt%)": "lignin_pct",
    "Pyrolysis temperature (°C)": "temperature_c",
    "HeatingRate(°C/min)": "heating_rate_c_min",
    "N2 flow rate (mL/min)": "n2_flow_ml_min",
    "ParticleSize(mm)": "particle_size_mm",
    "ParticleSize(μm)": "particle_size_um",
    "bio-liquid yield(wt%)": "bio_liquid_yield_pct",
}

NUMERIC_COLUMNS = list(COLUMN_MAP.values())
REQUIRED_SOURCE_COLUMNS = set(COLUMN_MAP)


def _read_text(path_or_buffer: str | Path | IO[bytes]) -> str:
    if isinstance(path_or_buffer, (str, Path)):
        return Path(path_or_buffer).read_text(encoding="utf-8-sig", errors="replace")

    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)

    if hasattr(path_or_buffer, "getvalue"):
        content = path_or_buffer.getvalue()
    else:
        content = path_or_buffer.read()

    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)

    if isinstance(content, bytes):
        return content.decode("utf-8-sig", errors="replace")
    return str(content)


def _find_header_row(lines: list[str]) -> int:
    required_hits = {"Cellulose(wt%)", "Hemicellulose(wt%)", "bio-liquid yield(wt%)"}
    for index, line in enumerate(lines):
        if required_hits.issubset(set(part.strip() for part in line.replace(";", ",").split(","))):
            return index

    for index, line in enumerate(lines):
        if "Cellulose" in line and "bio-liquid yield" in line:
            return index

    return 0


def _read_csv_flexibly(path_or_buffer: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read common CSV exports, including semicolon files with preamble rows."""
    text = _read_text(path_or_buffer)
    lines = text.splitlines()
    header_row = _find_header_row(lines)
    csv_text = "\n".join(lines[header_row:])
    header = lines[header_row] if lines else ""
    separator = ";" if header.count(";") > header.count(",") else ","

    return pd.read_csv(
        StringIO(csv_text),
        sep=separator,
        decimal="," if separator == ";" else ".",
        engine="python",
    )


def load_data(path_or_buffer: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load an experiment CSV and normalize it for analysis."""
    raw = _read_csv_flexibly(path_or_buffer)
    raw.columns = [str(column).strip() for column in raw.columns]

    missing = REQUIRED_SOURCE_COLUMNS - set(raw.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    df = raw.rename(columns=COLUMN_MAP).copy()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["particle_size_um_from_mm"] = df["particle_size_mm"] * 1000
    df["particle_size_um"] = np.where(
        df["particle_size_um"].notna(),
        df["particle_size_um"],
        df["particle_size_um_from_mm"],
    )

    both_present = df["particle_size_mm"].notna() & df["particle_size_um"].notna()
    mismatch = both_present & ~np.isclose(
        df["particle_size_um"],
        df["particle_size_um_from_mm"],
        rtol=0.02,
        atol=2,
        equal_nan=True,
    )
    df["particle_size_mismatch"] = mismatch

    clean_columns = [
        "cellulose_pct",
        "hemicellulose_pct",
        "lignin_pct",
        "temperature_c",
        "heating_rate_c_min",
        "n2_flow_ml_min",
        "particle_size_um",
        "bio_liquid_yield_pct",
        "particle_size_mismatch",
    ]
    df = df[clean_columns]

    critical = ["temperature_c", "heating_rate_c_min", "bio_liquid_yield_pct"]
    return df.dropna(subset=critical).reset_index(drop=True)
