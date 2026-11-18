"""Structured dataset exploration. Run: python -m src.eda
Writes results/inventory.csv, results/network_stats.csv, results/data_quality.txt and figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import DATA_DIR, FIG_DIR, RESULTS_DIR
from .io import list_trips, load_trip


def trip_row(part, tid):
    tr = load_trip(part, tid)
    d = Path_size(part, tid)
    t = tr.track
    dt = np.diff(t["t"].values)
    step = np.hypot(np.diff(t["x"].values), np.diff(t["y"].values))
    n_nodes, n_arcs = len(tr.nodes), len(tr.arcs)
    # connectivity (weak) of the road graph
    g = coo_matrix((np.ones(n_arcs), (tr.arcs.u, tr.arcs.v)), shape=(n_nodes, n_nodes))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    deg = np.bincount(np.r_[tr.arcs.u, tr.arcs.v], minlength=n_nodes)
    # route checks
    r = tr.route
    r_valid = bool(r is not None and r.min() >= 0 and r.max() < n_arcs)
    contiguous = np.nan
    if r_valid and len(r) > 1:
        v = tr.arcs.v.values[r[:-1]]
        u = tr.arcs.u.values[r[1:]]
        contiguous = float((v == u).mean())
    return dict(
        partition=part, trip=tid, crs=tr.crs, lon_c=float(t.lon.median()), lat_c=float(t.lat.median()),
        n_obs=len(t), n_route_arcs=0 if r is None else len(r),
        track_na=int(t.isna().sum().sum()),
        lon_out_of_range=int(((t.lon.abs() > 180) | (t.lat.abs() > 90)).sum()),
        t_monotonic=bool((dt > 0).all()), dt_eq_1s_pct=float((dt == 1).mean() * 100),
        dt_max=float(dt.max()), n_gaps_gt1s=int((dt > 1).sum()),
        track_len_m=float(step.sum()), max_step_m=float(step.max()),
        extent_x_m=float(np.ptp(t.x)), extent_y_m=float(np.ptp(t.y)),
        n_nodes=n_nodes, n_arcs=n_arcs, n_components=int(ncomp),
        largest_comp_pct=float(sizes.max() / n_nodes * 100),
        isolated_nodes=int((deg == 0).sum()), mean_degree=float(deg.mean()),
        route_valid=r_valid, route_contiguity=contiguous,
        size_kb=d,
    )


def Path_size(part, tid):
    p = DATA_DIR / part / tid
    return round(sum(f.stat().st_size for f in p.iterdir()) / 1024, 1)


def main():
    rows = []
    for part in ("train", "student_test"):
        for tid in list_trips(part):
            rows.append(trip_row(part, tid))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "inventory.csv", index=False)

    summ = df.groupby("partition").agg(
        trips=("trip", "count"), obs_total=("n_obs", "sum"), obs_mean=("n_obs", "mean"),
        obs_min=("n_obs", "min"), obs_max=("n_obs", "max"),
        len_km_mean=("track_len_m", lambda s: s.mean() / 1000),
        nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"),
        comps_mean=("n_components", "mean"), isolated_total=("isolated_nodes", "sum"),
        na_total=("track_na", "sum"),
    )
    summ.to_csv(RESULTS_DIR / "network_stats.csv")

    lines = ["DATA QUALITY SUMMARY", "=" * 20, summ.T.to_string(), ""]
    lines.append("Trips per metric CRS (UTM zone chosen from trip centroid):")
    lines.append(df.groupby(["partition", "crs"]).size().to_string())
    lines.append(f"Trips with non-monotonic timestamps: {int((~df.t_monotonic).sum())}")
    lines.append(f"Trips with missing values in .track: {int((df.track_na > 0).sum())}")
    lines.append(f"Trips with invalid route arc ids: {int((~df.route_valid).sum())}")
    lines.append(f"Route contiguity (v[i]==u[i+1]) mean: {df.route_contiguity.mean():.4f}; "
                 f"min: {df.route_contiguity.min():.4f}")
    lines.append(f"Mean % of 1 s steps: {df.dt_eq_1s_pct.mean():.2f}")
    lines.append("Trips with largest GPS time gap:")
    lines.append(df.nlargest(5, "dt_max")[["partition", "trip", "dt_max", "n_gaps_gt1s"]].to_string(index=False))
    lines.append("Trips with largest single-step jump (m):")
    lines.append(df.nlargest(5, "max_step_m")[["partition", "trip", "max_step_m"]].to_string(index=False))
    (RESULTS_DIR / "data_quality.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df.n_obs.hist(ax=ax[0], bins=20); ax[0].set_title("GPS observations per trip (train+student)")
    df.track_len_m.div(1000).hist(ax=ax[1], bins=20); ax[1].set_title("Track length (km)")
    df.n_arcs.hist(ax=ax[2], bins=20); ax[2].set_title("Road-network arcs per trip")
    plt.tight_layout(); plt.savefig(FIG_DIR / "eda_distributions.png", dpi=130); plt.close()

    # one trajectory over its road network (train partition)
    tid = list_trips("train")[0]
    tr = load_trip("train", tid)
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, xmax = tr.track.x.min() - 300, tr.track.x.max() + 300
    ymin, ymax = tr.track.y.min() - 300, tr.track.y.max() + 300
    nx, ny = tr.nodes.x.values, tr.nodes.y.values
    u, v = tr.arcs.u.values, tr.arcs.v.values
    keep = ((nx[u] > xmin) & (nx[u] < xmax) & (ny[u] > ymin) & (ny[u] < ymax))
    for a, b in zip(u[keep], v[keep]):
        ax.plot([nx[a], nx[b]], [ny[a], ny[b]], color="lightgray", lw=0.5)
    r = tr.route
    for a in r:
        ax.plot([nx[u[a]], nx[v[a]]], [ny[u[a]], ny[v[a]]], color="green", lw=2)
    ax.scatter(tr.track.x, tr.track.y, s=4, color="blue", zorder=3)
    ax.set_title(f"Trip {tid} (train): GPS (blue), network (grey), reference route (green)")
    ax.set_aspect("equal"); plt.savefig(FIG_DIR / f"eda_trip_{tid}.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
