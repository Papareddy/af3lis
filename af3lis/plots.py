"""Figures for an af3lis run.

Reads the ``metrics.tsv`` written by ``pipeline.py --collect`` and emits a
small, fixed set of publication-ready PNG/PDF panels. Handles BOTH aggregation
schemas: ``--agg flat`` (``<metric>_mean``) and ``--agg per_seed``
(``<metric>_mean_mean``), picking whichever is present.

Emitted into ``<figdir>``:

    peak_vs_ilis.png     PEAK vs iLIS for every chain pair, hit quadrant shaded
    ranked_hits.png      top pairs by iLIS, horizontal bars, hit bar marked
    heatmap_iLIS.png     chain-A x chain-B grid (grid-mode runs only)
    heatmap_PEAK.png     same, PEAK
    pae/<pair>.png       PAE matrix of each pair's top-ranked model

Colours come from a validated colourblind-safe categorical set; every figure
is legible in greyscale because identity is carried by position and direct
labels, never by hue alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURF = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9b9a95", "#e6e5e1"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c1272d"
GRAY = "#c9c8c3"
SEQ = LinearSegmentedColormap.from_list(
    "af3lis_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])

# house thresholds; overridable from the CLI
PEAK_BAR, ILIS_BAR = 0.70, 0.23


def _pick(df: pd.DataFrame, metric: str) -> str | None:
    """Column for `metric`, tolerating flat and per-seed schemas."""
    for cand in (f"{metric}_mean_mean", f"{metric}_mean", metric):
        if cand in df.columns:
            return cand
    return None


def _pair_label(row) -> str:
    """Readable pair name. Job names are '<a>___<b>' (AF3 lowercases them)."""
    n = str(row.get("name", ""))
    n = n.split("_20")[0]                      # drop an AF3 timestamp suffix
    if "___" in n:
        a, b = n.split("___", 1)
        return f"{a.upper()}·{b.upper()}"
    ci, cj = row.get("chain_i", ""), row.get("chain_j", "")
    return f"{n.upper()}[{ci}{cj}]"


def _ab(row) -> tuple[str, str] | None:
    n = str(row.get("name", "")).split("_20")[0]
    if "___" not in n:
        return None
    a, b = n.split("___", 1)
    return a.upper(), b.upper()


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#d9d8d4")


def _place_labels(ax, fig, items):
    """Greedy non-overlapping direct labels; leader line whenever displaced.

    matplotlib has no label-repulsion, and a scatter of chain pairs clusters
    hard in the top-right corner, so the naive fixed offset collides. Try a
    ring of candidate offsets and keep the first that misses everything placed
    so far; draw a thin leader back to the mark when the label had to move.
    """
    r = fig.canvas.get_renderer()
    placed = []
    cands = [(9, 4), (9, -12), (-11, 6), (-11, -14), (11, 20), (11, -26),
             (-13, 22), (-13, -30), (13, 36), (13, -42)]
    for x, y, txt in sorted(items, key=lambda t: -t[1]):
        for k, (dx, dy) in enumerate(cands):
            t = ax.annotate(txt, (x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=8.2, color=INK,
                            fontweight="bold", ha="left" if dx > 0 else "right",
                            va="center", zorder=7)
            bb = t.get_window_extent(renderer=r).expanded(1.06, 1.30)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                if k:
                    ax.annotate("", (x, y), xytext=(dx, dy),
                                textcoords="offset points", zorder=5,
                                arrowprops=dict(arrowstyle="-", color=MUTED,
                                                lw=.6, shrinkA=1, shrinkB=3))
                break
            t.remove()


def _save(fig, path, also_pdf=True):
    fig.savefig(path, dpi=190, facecolor=SURF)
    if also_pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor=SURF)
    plt.close(fig)
    print(f"[plots] {path}")


# ---------------------------------------------------------------------------

def peak_vs_ilis(df, figdir, peak_bar=PEAK_BAR, ilis_bar=ILIS_BAR, label_top=12):
    ic, pc = _pick(df, "iLIS"), _pick(df, "PEAK")
    if not ic or not pc:
        print("[plots] skip peak_vs_ilis: no iLIS/PEAK columns", file=sys.stderr)
        return
    d = df.dropna(subset=[ic, pc]).copy()
    if d.empty:
        print("[plots] skip peak_vs_ilis: no finite values", file=sys.stderr)
        return
    d["hit"] = (d[pc] >= peak_bar) & (d[ic] > ilis_bar)
    d["lab"] = d.apply(_pair_label, axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    xhi = max(1.02, float(d[pc].max()) * 1.05)
    yhi = max(0.35, float(d[ic].max()) * 1.15)
    ax.add_patch(plt.Rectangle((peak_bar, ilis_bar), xhi - peak_bar, yhi - ilis_bar,
                               facecolor=BLUE, alpha=.055, lw=0, zorder=0))
    ax.axvline(peak_bar, ls="--", lw=.9, color=MUTED)
    ax.axhline(ilis_bar, ls="--", lw=.9, color=MUTED)
    for hit, sub in d.groupby("hit"):
        ax.scatter(sub[pc], sub[ic], s=70 if hit else 42,
                   c=BLUE if hit else GRAY, edgecolor=SURF if hit else "#b4b3ae",
                   lw=1.3 if hit else .6, zorder=4 if hit else 2,
                   label=f"clears the bar (n={len(sub)})" if hit else f"below (n={len(sub)})")
    ax.set_xlim(min(0, float(d[pc].min()) - .03), xhi)
    ax.set_ylim(min(0, float(d[ic].min()) - .02), yhi)
    _place_labels(ax, fig, [(r[pc], r[ic], r["lab"])
                            for _, r in d[d.hit].nlargest(label_top, ic).iterrows()])
    ax.set_xlabel(f"PEAK   ({pc})", fontsize=10, color=INK2)
    ax.set_ylabel(f"iLIS   ({ic})", fontsize=10, color=INK2)
    ax.set_title(f"Interface confidence, {len(d)} chain pairs\n"
                 f"shaded = PEAK ≥ {peak_bar} AND iLIS > {ilis_bar}",
                 fontsize=11.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    _save(fig, os.path.join(figdir, "peak_vs_ilis.png"))


def ranked_hits(df, figdir, top=30, peak_bar=PEAK_BAR, ilis_bar=ILIS_BAR):
    ic, pc = _pick(df, "iLIS"), _pick(df, "PEAK")
    if not ic:
        return
    d = df.dropna(subset=[ic]).copy()
    if d.empty:
        return
    d["lab"] = d.apply(_pair_label, axis=1)
    d = d.nlargest(min(top, len(d)), ic).iloc[::-1]
    hit = (d[pc] >= peak_bar) & (d[ic] > ilis_bar) if pc else d[ic] > ilis_bar
    fig, ax = plt.subplots(figsize=(8.8, max(3.2, .30 * len(d) + 1.6)))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    ax.grid(False, axis="y")
    ax.barh(np.arange(len(d)), d[ic], color=[BLUE if h else GRAY for h in hit],
            edgecolor="k", lw=.5, height=.72)
    ax.axvline(ilis_bar, ls="--", lw=1.1, color=RED)
    ax.text(ilis_bar, len(d) - .2, f" iLIS bar {ilis_bar}", fontsize=8.4, color=RED)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["lab"], fontsize=8.2)
    for t, h in zip(ax.get_yticklabels(), hit):
        t.set_color(INK if h else INK2)
        if h:
            t.set_fontweight("bold")
    ax.set_xlabel(f"iLIS   ({ic})", fontsize=10, color=INK2)
    ax.set_title(f"Top {len(d)} chain pairs by iLIS", fontsize=11.5, color=INK, pad=9)
    fig.tight_layout()
    _save(fig, os.path.join(figdir, "ranked_hits.png"))


def heatmaps(df, figdir, metrics=("iLIS", "PEAK")):
    """chain-A x chain-B grids. Silently skipped when the run is not a grid."""
    ab = df.apply(_ab, axis=1)
    if ab.isna().all():
        return
    d = df.loc[ab.notna()].copy()
    d["A"] = [x[0] for x in ab[ab.notna()]]
    d["B"] = [x[1] for x in ab[ab.notna()]]
    if d.A.nunique() < 2 and d.B.nunique() < 2:
        return
    for m in metrics:
        col = _pick(d, m)
        if not col:
            continue
        piv = d.pivot_table(index="A", columns="B", values=col, aggfunc="max")
        if piv.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(max(5.0, .62 * piv.shape[1] + 3.0),
                                        max(3.6, .44 * piv.shape[0] + 2.2)))
        fig.patch.set_facecolor(SURF)
        ax.set_facecolor(SURF)
        im = ax.imshow(piv.values, cmap=SEQ, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8.4)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=8.4)
        vmid = np.nanmean(piv.values)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                            color="white" if v > vmid else INK)
        ax.set_xlabel("chain B", fontsize=10, color=INK2)
        ax.set_ylabel("chain A", fontsize=10, color=INK2)
        ax.set_title(f"{m} per chain pair   ({col})", fontsize=11.5, color=INK, pad=9)
        fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
        for s in ax.spines.values():
            s.set_color("#d9d8d4")
        fig.tight_layout()
        _save(fig, os.path.join(figdir, f"heatmap_{m}.png"))


def pae_panels(out_dir, figdir, max_pairs=60):
    """PAE matrix of each pair's top-ranked model, with chain blocks marked."""
    try:
        from af3lis import af3_io
    except Exception as e:                                  # pragma: no cover
        print(f"[plots] skip PAE panels ({e})", file=sys.stderr)
        return
    pdir = os.path.join(figdir, "pae")
    os.makedirs(pdir, exist_ok=True)
    n = 0
    for job in af3_io.iter_jobs(out_dir):
        if n >= max_pairs:
            print(f"[plots] PAE panels capped at {max_pairs}", file=sys.stderr)
            break
        try:
            fr = af3_io.load_best(job)
        except Exception as e:
            print(f"[plots] skip PAE {os.path.basename(job)}: {e}", file=sys.stderr)
            continue
        pae, labels = fr.pae, list(fr.token_chain_ids)
        fig, ax = plt.subplots(figsize=(5.6, 4.9))
        fig.patch.set_facecolor(SURF)
        ax.set_facecolor(SURF)
        im = ax.imshow(pae, cmap="viridis_r", vmin=0, vmax=31, aspect="equal")
        # chain boundaries in token space
        bounds, seen = [], None
        for i, c in enumerate(labels):
            if c != seen:
                if seen is not None:
                    bounds.append(i)
                seen = c
        for b in bounds:
            ax.axhline(b - .5, color="w", lw=1.1)
            ax.axvline(b - .5, color="w", lw=1.1)
        order, mid = [], []
        for c in dict.fromkeys(labels):
            idx = [i for i, x in enumerate(labels) if x == c]
            order.append(c)
            mid.append(np.mean(idx))
        ax.set_xticks(mid); ax.set_xticklabels(order, fontsize=9)
        ax.set_yticks(mid); ax.set_yticklabels(order, fontsize=9)
        nm = os.path.basename(job).split("_20")[0]
        ax.set_title(f"{nm}\nPAE, top model (seed {fr.seed} sample {fr.sample})",
                     fontsize=10, color=INK, pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
        cb.set_label("expected position error (Å)", fontsize=8.6, color=INK2)
        fig.tight_layout()
        _save(fig, os.path.join(pdir, f"{nm}.png"), also_pdf=False)
        n += 1


def make_all(metrics_tsv, out_dir=None, figdir=None,
             peak_bar=PEAK_BAR, ilis_bar=ILIS_BAR, with_pae=True):
    """Every figure for one run. `out_dir` is the AF3 output root (for PAE)."""
    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(metrics_tsv)), "figures")
    os.makedirs(figdir, exist_ok=True)
    df = pd.read_csv(metrics_tsv, sep="\t")
    print(f"[plots] {len(df)} rows from {metrics_tsv} -> {figdir}")
    peak_vs_ilis(df, figdir, peak_bar, ilis_bar)
    ranked_hits(df, figdir, peak_bar=peak_bar, ilis_bar=ilis_bar)
    heatmaps(df, figdir)
    if with_pae and out_dir and os.path.isdir(out_dir):
        pae_panels(out_dir, figdir)
    return figdir


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Plot an af3lis metrics.tsv.")
    ap.add_argument("metrics_tsv")
    ap.add_argument("--out-dir", default=None,
                    help="AF3 output root (out/ or out_pack/) for PAE panels")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--peak-bar", type=float, default=PEAK_BAR)
    ap.add_argument("--ilis-bar", type=float, default=ILIS_BAR)
    ap.add_argument("--no-pae", action="store_true")
    a = ap.parse_args(argv)
    make_all(a.metrics_tsv, a.out_dir, a.figdir, a.peak_bar, a.ilis_bar,
             with_pae=not a.no_pae)


if __name__ == "__main__":
    main()
