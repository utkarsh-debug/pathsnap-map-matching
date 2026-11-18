"""Raw-data readers for .track / .nodes / .arcs / .route files."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import DATA_DIR


def utm_epsg(lon: float, lat: float) -> str:
    """UTM CRS for a location (trips come from different regions, e.g. Germany and Ecuador)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, t, x, y (metres)
    nodes: pd.DataFrame          # lon, lat, x, y ; row index = node_id
    arcs: pd.DataFrame           # u, v ; row index = arc_id
    route: np.ndarray | None     # ground-truth arc_ids (None if absent)
    crs: str = ""                # metric CRS chosen for this trip


def list_trips(partition: str, data_dir: Path = DATA_DIR) -> list[str]:
    p = Path(data_dir) / partition
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def _read(path: Path, names, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(partition: str, trip_id: str, data_dir: Path = DATA_DIR,
              with_route: bool = True) -> Trip:
    """Load one trip. with_route=False keeps labels out of the pipeline (API inference)."""
    d = Path(data_dir) / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"])
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"])
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], dtype=np.int64)
    crs = utm_epsg(track["lon"].median(), track["lat"].median())
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    track["x"], track["y"] = tf.transform(track["lon"].values, track["lat"].values)
    nodes["x"], nodes["y"] = tf.transform(nodes["lon"].values, nodes["lat"].values)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = _read(rp, ["arc"], dtype=np.int64)["arc"].values
    return Trip(trip_id, partition, track, nodes, arcs, route, crs)
