# load.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

PathLike = Union[str, Path]

@dataclass(frozen=True)
class CMAPSSSchema:
    """
    CMAPSS files have 26 whitespace-separated columns:
    unit, cycle, 3 operational settings, 21 sensor measurements. [web:40][web:47]
    """
    unit_col: str = "unit"
    cycle_col: str = "cycle"
    setting_cols: tuple[str, ...] = ("setting1", "setting2", "setting3")
    sensor_cols: tuple[str, ...] = tuple(f"s{i}" for i in range(1, 22))

    @property
    def columns(self) -> list[str]:
        return [self.unit_col, self.cycle_col, *self.setting_cols, *self.sensor_cols]

SCHEMA = CMAPSSSchema()


def read_cmapss_txt(path: PathLike, *, schema: CMAPSSSchema = SCHEMA) -> pd.DataFrame:
    """
    Robust reader for CMAPSS train_FD00x.txt / test_FD00x.txt.
    Handles trailing whitespace that can create an extra empty column. [web:40]
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        delim_whitespace=True,
        header=None,
        engine="python",
    )

    # Drop any all-NaN columns (common when file has trailing spaces)
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 26:
        raise ValueError(f"{path.name}: expected 26 columns, got {df.shape[1]}")

    df.columns = schema.columns

    # Types + sort for safety
    df[schema.unit_col] = df[schema.unit_col].astype(int)
    df[schema.cycle_col] = df[schema.cycle_col].astype(int)
    df = df.sort_values([schema.unit_col, schema.cycle_col]).reset_index(drop=True)

    return df


def sanity_check_cycles(
    df: pd.DataFrame,
    *,
    unit_col: str = SCHEMA.unit_col,
    cycle_col: str = SCHEMA.cycle_col,
) -> None:
    """
    Checks that cycles are strictly increasing within each unit.
    Raises AssertionError if not. (Helpful before windowing.) [web:40]
    """
    bad_units = []
    for unit, g in df.groupby(unit_col, sort=False):
        cyc = g[cycle_col].to_numpy()
        if len(cyc) == 0:
            continue
        if (cyc[1:] <= cyc[:-1]).any():
            bad_units.append(unit)

    assert not bad_units, f"Non-increasing cycles found for units: {bad_units[:10]}"


def load_fd001(
    data_dir: PathLike = "data/raw",
    *,
    schema: CMAPSSSchema = SCHEMA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader for FD001.
    Returns: (train_df, test_df). [web:40][file:1]
    """
    data_dir = Path(data_dir)
    train_df = read_cmapss_txt(data_dir / "train_FD001.txt", schema=schema)
    test_df  = read_cmapss_txt(data_dir / "test_FD001.txt", schema=schema)

    sanity_check_cycles(train_df, unit_col=schema.unit_col, cycle_col=schema.cycle_col)
    sanity_check_cycles(test_df, unit_col=schema.unit_col, cycle_col=schema.cycle_col)

    return train_df, test_df
