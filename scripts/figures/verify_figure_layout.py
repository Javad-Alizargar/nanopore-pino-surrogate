"""TASK 5B.2 — programmatic pixel bounding-box verification of Figures 1-6.

Renders each figure at its real DPI and inspects artist
``get_window_extent(renderer)`` in pixels:

  1. Box overflow  — any Text placed inside a Rectangle / FancyBboxPatch (or
     carrying its own bbox patch) must lie inside that patch with >= 6 px pad.
  2. Text overlap  — no two *visible* Text artists from different sources may
     overlap (area > tolerance).  Tick labels of the same axis are ignored.
  3. Saved-PNG margin — the rendered PNG must have >= 8 px of pure-white margin
     on every side (catches any clipped / edge-touching content).

Outputs ``results/figure_data/figure_layout_report.json``; exits non-zero if any
figure fails.  Local only; no paid calls.

    .venv/bin/python scripts/figures/verify_figure_layout.py [fig1 ... ]
"""

import importlib
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import image as mpimg  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402

PAD_REQUIRED = 6.0     # px padding text must keep inside its box
OVERLAP_TOL = 12.0     # px^2 overlap area tolerated between two texts
PNG_MARGIN = 8.0       # px of white margin required around saved PNG
PANEL_GAP = 14.0       # px minimum clearance for every panel letter

FIGURES = [
    "fig1_workflow_and_architecture",
    "fig2_benchmark_and_field_reconstruction",
    "fig3_h3_charge_reparameterization_ablation",
    "fig4_h1_h2_interp_vs_ood_regimes",
    "fig5_parametric_physics_sweeps",
    "fig6_h4_timing_and_amortization_breakeven",
]
PANEL_SPECS = {
    "fig1_workflow_and_architecture": set("ABCD"),
    "fig2_benchmark_and_field_reconstruction": set("ABCDEF"),
    "fig3_h3_charge_reparameterization_ablation": set("ABCDEF"),
    "fig4_h1_h2_interp_vs_ood_regimes": set("ABCD"),
    "fig5_parametric_physics_sweeps": set("ABC"),
    "fig6_h4_timing_and_amortization_breakeven": set("ABCD"),
}

captured = {}
_orig_save = K.save_fig


def _capture(fig, name, data=None):
    captured[name] = fig
    _orig_save(fig, name, data)


def _finite(bb):
    if bb is None:
        return False
    return abs(bb.x0) != float("inf") and bb.x0 == bb.x0


def _extent(a, rend):
    try:
        bb = a.get_window_extent(rend)
        return bb if _finite(bb) else None
    except Exception:
        return None


def _overlap_area(a, b):
    x0 = max(a.x0, b.x0); x1 = min(a.x1, b.x1)
    y0 = max(a.y0, b.y0); y1 = min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _rect_gap(a, b):
    """Euclidean gap in px between two axis-aligned bounding boxes."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return float(np.hypot(dx, dy))


def _in_view_ticks(axis, lo, hi):
    lo, hi = min(lo, hi), max(lo, hi)
    out = []
    for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
        try:
            loc = tk.get_loc()
        except Exception:
            continue
        if loc is None or not (lo - 1e-12 <= loc <= hi + 1e-12):
            continue
        out.append(tk.label1)
    return out


def _collect(fig, rend):
    """Return list of visible texts that are actually drawn in-view."""
    items = []
    for ax in fig.get_axes():
        axis_on = ax.axison
        for t in ax.texts:
            items.append((t, "annotation", ax))
        items.append((ax.title, "title", ax))
        items.append((ax.xaxis.label, "label", ax))
        items.append((ax.yaxis.label, "label", ax))
        if axis_on:
            xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
            items.extend((t, "tick", ax) for t in _in_view_ticks(ax.xaxis, xlo, xhi))
            items.extend((t, "tick", ax) for t in _in_view_ticks(ax.yaxis, ylo, yhi))
        leg = ax.get_legend()
        if leg is not None:
            items.extend((t, "legend", ax) for t in leg.get_texts())
    for t in fig.texts:
        items.append((t, "figtext", None))
    out = []
    for t, kind, ax in items:
        if not t.get_visible() or t.get_text().strip() == "":
            continue
        e = _extent(t, rend)
        if e is None:
            continue
        out.append({"t": t, "e": e, "kind": kind, "ax": ax, "txt": t.get_text()})
    # dedup by id
    seen = set(); uniq = []
    for d in out:
        if id(d["t"]) in seen:
            continue
        seen.add(id(d["t"])); uniq.append(d)
    return uniq


def check_figure(name, fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rend = canvas.get_renderer()
    issues = []

    texts = _collect(fig, rend)

    # 1. box overflow
    patches = []
    for ax in fig.get_axes():
        for p in ax.patches:
            if isinstance(p, (Rectangle, FancyBboxPatch)):
                pe = _extent(p, rend)
                if pe is not None:
                    patches.append(pe)
    for d in texts:
        if d["kind"] != "annotation":
            continue
        te = d["e"]; t = d["t"]
        bp = t.get_bbox_patch()
        if bp is not None:
            be = _extent(bp, rend)
            if be is not None:
                pad = min(te.x0 - be.x0, be.x1 - te.x1, te.y0 - be.y0, be.y1 - te.y1)
                if pad < PAD_REQUIRED:
                    issues.append({"type": "box_padding", "text": t.get_text()[:40],
                                   "pad_px": round(float(pad), 1)})
        cx, cy = 0.5 * (te.x0 + te.x1), 0.5 * (te.y0 + te.y1)
        for pe in patches:
            if pe.x0 <= cx <= pe.x1 and pe.y0 <= cy <= pe.y1:
                pad = min(te.x0 - pe.x0, pe.x1 - te.x1, te.y0 - pe.y0, pe.y1 - te.y1)
                if pad < PAD_REQUIRED:
                    issues.append({"type": "text_out_of_box", "text": t.get_text()[:40],
                                   "pad_px": round(float(pad), 1)})

    # 2. panel labels: exact set, inside figure, and >= PANEL_GAP px clearance
    expected = PANEL_SPECS.get(name, set())
    panel = [d for d in texts if re.fullmatch(r"[A-F]", d["txt"].strip())]
    found = [d["txt"].strip() for d in panel]
    panel_issue_start = len(issues)
    for letter in sorted(expected - set(found)):
        issues.append({"type": "panel_missing", "letter": letter})
    for letter in sorted(set(found) - expected):
        issues.append({"type": "panel_extra", "letter": letter})
    for letter in sorted({x for x in found if found.count(x) > 1}):
        issues.append({"type": "panel_duplicate", "letter": letter})
    fig_bb = fig.bbox
    min_gap = None
    for d in panel:
        e = d["e"]
        inside = (e.x0 >= fig_bb.x0 and e.x1 <= fig_bb.x1
                  and e.y0 >= fig_bb.y0 and e.y1 <= fig_bb.y1)
        if not inside:
            issues.append({"type": "panel_out_of_bounds", "letter": d["txt"],
                           "bbox": [round(e.x0, 1), round(e.y0, 1),
                                    round(e.x1, 1), round(e.y1, 1)]})
        gaps = [(_rect_gap(e, o["e"]), f'{o["kind"]}:{o["txt"][:28]}')
                for o in texts if o is not d]
        for pi, p in enumerate(patches):
            gaps.append((_rect_gap(e, p), f"box{pi}"))
        local, nearest = min(gaps) if gaps else (0.0, "-")
        min_gap = local if min_gap is None else min(min_gap, local)
        if local < PANEL_GAP:
            issues.append({"type": "panel_clearance", "letter": d["txt"],
                           "gap_px": round(local, 1), "required_px": PANEL_GAP,
                           "nearest": nearest})
    panel_ok = len(issues) == panel_issue_start and set(found) == expected
    panel_info = {"expected": "".join(sorted(expected)), "found": "".join(sorted(found)),
                  "set_ok": set(found) == expected, "all_in_bounds": not any(
                      i["type"] == "panel_out_of_bounds" for i in issues[panel_issue_start:]),
                  "min_gap_px": None if min_gap is None else round(min_gap, 1),
                  "ok": bool(panel_ok and (min_gap is None or min_gap >= PANEL_GAP))}

    # 3. text overlap (skip same-axis tick pairs)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if a["kind"] == "tick" and b["kind"] == "tick" and a["ax"] is b["ax"]:
                continue
            area = _overlap_area(a["e"], b["e"])
            if area > OVERLAP_TOL:
                issues.append({"type": "text_overlap", "area_px2": round(float(area), 1),
                               "a": a["txt"][:30], "b": b["txt"][:30],
                               "kinds": [a["kind"], b["kind"]],
                               "ax": [a["ax"].get_title()[:18] if a["ax"] else "-",
                                      b["ax"].get_title()[:18] if b["ax"] else "-"],
                               "c": [[round(0.5*(a["e"].x0+a["e"].x1)), round(0.5*(a["e"].y0+a["e"].y1))],
                                     [round(0.5*(b["e"].x0+b["e"].x1)), round(0.5*(b["e"].y0+b["e"].y1))]]})

    # 4. saved-PNG white margin
    png = os.path.join(K.FIGDIR, name + ".png")
    if os.path.exists(png):
        img = mpimg.imread(png)
        rgb = img[..., :3]
        nonwhite = (rgb < 0.96).any(axis=-1)
        if nonwhite.any():
            ys, xs = np.where(nonwhite)
            H, W = nonwhite.shape
            margins = {"top": int(ys.min()), "bottom": int(H - 1 - ys.max()),
                       "left": int(xs.min()), "right": int(W - 1 - xs.max())}
            bad = {k: v for k, v in margins.items() if v < PNG_MARGIN}
            if bad:
                issues.append({"type": "png_margin", "margins": margins,
                               "required": PNG_MARGIN, "size": [W, H]})
    return issues, panel_info


def main():
    which = sys.argv[1:] or FIGURES
    K.save_fig = _capture
    report = {"pad_required_px": PAD_REQUIRED, "overlap_tol_px2": OVERLAP_TOL,
              "png_margin_px": PNG_MARGIN, "panel_gap_px": PANEL_GAP,
              "figures": {}}
    for mod in which:
        importlib.import_module(mod)
    total = 0
    for name in list(captured.keys()):
        fig = captured[name]
        issues, panel_info = check_figure(name, fig)
        plt.close(fig)
        report["figures"][name] = {"n_issues": len(issues), "issues": issues,
                                   "panel_labels": panel_info}
        total += len(issues)
        print(f"[layout] {name}: {'OK' if not issues else str(len(issues)) + ' ISSUES'}")
        print(f"           panel labels: {panel_info}")
        for it in issues[:40]:
            print("        ", it)
    report["total_issues"] = total
    out = os.path.join(K.FIGDATA, "figure_layout_report.json")
    os.makedirs(K.FIGDATA, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[layout] total issues = {total} -> {os.path.relpath(out, K.ROOT)}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
