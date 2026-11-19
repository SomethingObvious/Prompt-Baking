"""Puts every run in a results folder on one HTML page: settings, KL curves and test scores."""

import argparse
import html
import json
import math
import re
import webbrowser
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.offline import get_plotlyjs

KL_LINE = re.compile(r"Epoch (\d+) has a train KL of (\S+) and a validation KL of (\S+)")
MATH_EPOCH = re.compile(r"mathtest_ep(\d+)_")

# The same four colours on every chart, so a bar means the same thing everywhere.
CONDITIONS = {
    "base_nosys": ("Base, No Prompt", "#9aa0a6"),
    "base_sys": ("Base With Prompt", "#4c78a8"),
    "peft_nosys": ("Baked, No Prompt", "#f58518"),
    "peft_sys": ("Baked With Prompt", "#54a24b"),
}
# test_math_model.py calls the base model with the prompt "base" and the baked one "peft".
MATH_KINDS = {"base": "base_sys", "peft": "peft_nosys", "peft_sys": "peft_sys"}

STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 24px;
       color: #202124; background: #fff; }
h1 { margin-bottom: 0; }
h2 { margin-top: 40px; border-bottom: 1px solid #dadce0; padding-bottom: 4px; }
.note { color: #5f6368; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
th { color: #5f6368; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.empty { color: #5f6368; font-style: italic; }
"""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_kl(run_dir):
    """{epoch: (train KL, val KL)} from train_loop.log, the last line winning after a resume."""
    log = run_dir / "train_loop.log"
    kl = {}
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if found := KL_LINE.search(line):
                kl[int(found[1])] = (float(found[2]), float(found[3]))
    return dict(sorted(kl.items()))


def epoch_order(epoch):
    """Epochs in number order, with an old result that only says "last" at the end."""
    return epoch if isinstance(epoch, int) else math.inf


def standard_error(data, key):
    """The standard error of a mean score, from its spread and how many questions it took."""
    spread, scores = data.get(f"std_eval_{key}"), data.get(f"eval_{key}")
    if spread is None or not scores:
        return None
    return spread / math.sqrt(len(scores))


def read_run(root, run_dir):
    instruct = []
    for path in sorted(run_dir.glob("dataset_*_epoch_*_numquestions_*.json")):
        data = read_json(path)
        args = data.get("args", {})
        instruct.append(
            {
                "epoch": data.get("epoch", args.get("model_epoch")),
                "prompt": Path(args.get("u_file") or "").stem.removesuffix("_x0"),
                "scores": {k: data.get(f"mean_eval_{k}") for k in CONDITIONS},
                "spread": {k: standard_error(data, k) for k in CONDITIONS},
            }
        )
    instruct.sort(key=lambda t: (t["prompt"], epoch_order(t["epoch"])))
    maths = []
    for path in sorted(run_dir.glob("mathtest_ep*.json")):
        data = read_json(path)
        found = MATH_EPOCH.match(path.name)
        maths.append(
            {
                "epoch": data.get("epoch", int(found[1]) if found else None),
                "dataset": data.get("args", {}).get("dataset")
                or path.stem.split("datasetname")[-1],
                "scores": {MATH_KINDS[k]: data.get(f"mean_accuracy_{k}") for k in MATH_KINDS},
            }
        )
    maths.sort(key=lambda m: (m["dataset"], epoch_order(m["epoch"])))
    best = run_dir / "best.json"
    return {
        "name": run_dir.relative_to(root).as_posix() if run_dir != root else run_dir.name,
        "args": read_json(run_dir / "args.json"),
        "kl": read_kl(run_dir),
        "best": read_json(best) if best.is_file() else None,
        "instruct": instruct,
        "math": maths,
    }


def find_runs(root):
    """Every folder under root with an args.json in it, which is what training leaves."""
    return [read_run(root, p.parent) for p in sorted(root.rglob("args.json"))]


def runs_table(runs):
    rows = []
    for run in runs:
        args, best = run["args"], run["best"]
        done = max(run["kl"], default=-1) + 1
        text = [run["name"], args.get("model_name", ""), Path(args.get("data_path") or "").name]
        numbers = [
            args.get("r", ""),
            args.get("learning_rate", ""),
            f"{done} of {args.get('num_epochs', '?')}",
            best["epoch"] if best else "",
            f"{best['val_kl']:.4f}" if best else "",
        ]
        cells = [f"<td>{html.escape(str(t))}</td>" for t in text]
        cells += [f'<td class="num">{html.escape(str(n))}</td>' for n in numbers]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = ["Run", "Model", "Trajectories", "Rank", "Learning Rate", "Epochs", "Best Epoch"]
    header = "".join(f"<th>{h}</th>" for h in [*head, "Best Val KL"])
    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def kl_chart(runs):
    fig = go.Figure()
    palette = qualitative.Plotly
    for n, run in enumerate(r for r in runs if r["kl"]):
        epochs = list(run["kl"])
        train, val = zip(*run["kl"].values(), strict=True)
        group, colour = run["name"], palette[n % len(palette)]
        fig.add_scatter(
            x=epochs,
            y=val,
            name=f"{group} validation",
            legendgroup=group,
            mode="lines+markers",
            line={"color": colour},
            marker={"size": 5},
        )
        fig.add_scatter(
            x=epochs,
            y=train,
            name=f"{group} train",
            legendgroup=group,
            mode="lines",
            line={"dash": "dot", "color": colour},
        )
        if run["best"] and run["best"]["epoch"] in run["kl"]:
            fig.add_scatter(
                x=[run["best"]["epoch"]],
                y=[run["best"]["val_kl"]],
                name=f"{group} best",
                legendgroup=group,
                mode="markers",
                marker={"symbol": "star", "size": 14, "color": colour},
                showlegend=False,
            )
    fig.update_layout(xaxis_title="Epoch", yaxis_title="KL Divergence")
    # 1, 2 and 5 of each decade written out in full. Plotly's own log labels say "5" for 0.5.
    ticks = [m * 10.0**e for e in range(-4, 3) for m in (1, 2, 5)]
    fig.update_yaxes(type="log", tickvals=ticks, tickformat="~g")
    return fig if fig.data else None


def bar_chart(groups, conditions, y_title, percent):
    """One bar per condition for each labelled group of scores."""
    fig = go.Figure()
    labels = [label for label, _, _ in groups]
    for key in conditions:
        name, colour = CONDITIONS[key]
        errors = [(spread or {}).get(key) for _, _, spread in groups]
        fig.add_bar(
            name=name,
            x=labels,
            y=[scores.get(key) for _, scores, _ in groups],
            # Without any, plotly still draws an empty cap on every bar.
            error_y={"array": errors} if any(e is not None for e in errors) else None,
            marker_color=colour,
        )
    # Every scorer and every accuracy runs from 0 to 1.
    fig.update_layout(barmode="group", yaxis_title=y_title, yaxis_range=[0, 1])
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return fig if groups else None


def section(title, note, fig, empty):
    if fig is not None:
        fig.update_layout(template="plotly_white", margin={"t": 20, "b": 50})
    body = (
        fig.to_html(full_html=False, include_plotlyjs=False, default_height=460)
        if fig is not None
        else f'<p class="empty">{empty}</p>'
    )
    return f'<h2>{title}</h2><p class="note">{note}</p>{body}'


def build_html(root):
    root = Path(root)
    runs = find_runs(root)
    instruct = [
        (f"{r['name']}<br>{t['prompt']}, epoch {t['epoch']}", t["scores"], t["spread"])
        for r in runs
        for t in r["instruct"]
    ]
    math = [
        (f"{r['name']}<br>{m['dataset']}, epoch {m['epoch']}", m["scores"], None)
        for r in runs
        for m in r["math"]
    ]
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Prompt Baking Runs</title>",
        f"<style>{STYLE}</style><script>{get_plotlyjs()}</script></head><body>",
        "<h1>Prompt Baking Runs</h1>",
        f'<p class="note">{len(runs)} run{"" if len(runs) == 1 else "s"} in '
        f"{html.escape(str(root))}, read {when}.</p>",
        "<h2>Runs</h2>",
        runs_table(runs)
        if runs
        else f'<p class="empty">There are no runs in {html.escape(str(root))} yet. '
        "Train one with train_loop_custom.py and its folder shows up here.</p>",
        section(
            "KL Divergence",
            "How far the baked model without the prompt is from the base model with it, so lower "
            "is better. Solid lines are validation, dotted are training, and the star is the epoch "
            "kept as the best.",
            kl_chart(runs),
            "No run has logged an epoch yet.",
        ),
        section(
            "Prompt Following",
            "Scores from test_instruct_model.py. The orange bar is the one that matters, and a "
            "well baked prompt brings it up to the blue one. The whiskers are the standard error "
            "over the questions.",
            bar_chart(instruct, list(CONDITIONS), "Score", percent=False),
            "No prompt-following scores yet. Run test_instruct_model.py on a run to add them.",
        ),
        section(
            "Math Accuracy",
            "Exact-match accuracy from test_math_model.py.",
            bar_chart(math, list(MATH_KINDS.values()), "Accuracy", percent=True),
            "No math scores yet. Run test_math_model.py on a run to add them.",
        ),
        "</body></html>",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument(
        "--out",
        help="Where to write the page. It defaults to dashboard.html in the results folder.",
    )
    parser.add_argument("--open", action="store_true", help="Open the page in a browser.")
    args = parser.parse_args()
    root = Path(args.results_dir)
    if not root.is_dir():
        raise SystemExit(f"{root} isn't a folder.")
    out = Path(args.out) if args.out else root / "dashboard.html"
    out.write_text(build_html(root), encoding="utf-8")
    print(f"Wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
