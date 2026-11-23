"""Animation of a flume run: the bed and the water surface along the flume while the
deposit grows.

    python animate_flume.py results/joint_train26_variable_power_reduced_direct_hnet.json --run 2

The run is repeated with the fitted parameters of one closure of a results file of
``learn_flume_joint.py`` (``--stage``; default the learned hiding function where the file
has one, else the published closures), and the bed and the water surface after every step
are kept in ``results/<name>_run<run>_<stage>_history.npz`` (used again unless
``--rerun``).

Each frame shows the fixed floor of the flume, the deposit, the water and the bed surveys
made up to that time (the latest one in black, the earlier ones in grey). Only the bed
was surveyed; the water surface is that of the model.

Writes ``results/<name>_run<run>_<stage>.gif``, the same as ``.mp4`` where ``ffmpeg`` is
on the path (on Roar: ``module load ffmpeg``; H.264 where that build has the encoder,
else MPEG-4), and the last frame as ``.png``.

Units: SI; time in hours.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from plot_flume_joint import HIDING_STAGES, INK, LABELS, PUBLISHED_HIDING, profiles_of

WATER = "#cfe4f5"
WATER_LINE = "#2a78d6"
DEPOSIT = "#dcc19a"
BED_LINE = "#8a5a2b"
FLOOR = "#c4c4c4"
EARLIER = "0.55"                # grey of the surveys before the latest one
SOURCES = {1: "Seal et al. (1997)", 2: "Seal et al. (1997)", 3: "Seal et al. (1997)", 6: "Toro-Escobar et al. (2000)"}
HOLD_SECONDS = 2.0              # the last frame is held this long
LEGEND_LEFT = 0.66              # left edge of the legend (fraction of the axis width)


def history(path: Path, run: int, stage: str, rerun: bool) -> dict:
    """Bed and water surface of a run after every step, with the parameters of a stage.

    Returns:
        ``x``: sections (m from the feed point), shape [N]; ``floor``: fixed floor (m),
        shape [N]; ``hours``: time at the end of each step (h), shape [T]; ``bed`` and
        ``surface``: elevation of the bed and of the water surface (m), shape [T, N].
    """
    target = path.with_name(f"{path.stem}_run{run}_{stage}_history.npz")
    if target.exists() and not rerun:
        return dict(np.load(target))
    import jax

    import flume_case as flume
    import learn_flume as lf
    import learn_flume_joint as lj

    out = json.loads(path.read_text())
    case = flume.build(run, "wilcock_crowe", "unsteady", "bedload_mix", resistance=out["resistance"],
                       transport_stress=out["stress"], roughness_scale=out.get("roughness_scale", "length"))
    case = lj.with_closure(case, stage)
    free = lj.as_arrays(out["stages"][stage]["free"])
    result = jax.jit(lambda f: flume.run_case(case, lf.parameters(case, f)))(free)
    values = {
        "x": np.asarray(case.x), "floor": np.min(np.asarray(case.bed0.z), axis=1),
        "hours": np.asarray(result.time_end) / 3600.0, "bed": np.asarray(result.thalweg),
        "surface": np.asarray(result.eta),
    }
    np.savez_compressed(target, **values)
    return values


def movie_writer(fps: int):
    """Writer of the ``.mp4`` file, or None without ``ffmpeg``."""
    if shutil.which("ffmpeg") is None:
        return None
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    if " libx264 " in listed:
        return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    return animation.FFMpegWriter(fps=fps, codec="mpeg4", extra_args=["-pix_fmt", "yuv420p", "-q:v", "2"])


def draw(values: dict, run: int, label: str, minutes: float):
    """Figure of the animation and its frames.

    Returns:
        Tuple ``(fig, frames, update)``: the figure, the steps to show, and the function
        that draws the state after a step.
    """
    x, floor, hours, bed, surface = (values[key] for key in ("x", "floor", "hours", "bed", "surface"))
    step = max(1, round(minutes / 60.0 / float(hours[0])))
    frames = list(range(step - 1, hours.size, step))
    if frames[-1] != hours.size - 1:
        frames.append(hours.size - 1)
    surveys = {t: (stations, np.interp(stations, x, floor) + thickness) for t, (stations, thickness) in profiles_of(run).items()}

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    low = float(floor.min()) - 0.04
    top = 1.22 * float(surface.max())
    ax.fill_between(x, low, floor, color=FLOOR, lw=0)
    ax.plot(x, floor, color=INK, lw=1.2)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(low, top)
    ax.set_xlabel("distance from the feed point (m)")
    ax.set_ylabel("elevation (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Run {run} of {SOURCES[run]}: {label}", loc="left", fontsize=14)
    clock = ax.text(1.0, 1.02, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=14, weight="bold", color=INK)
    span, height = float(x[-1] - x[0]), top - low
    arrow = {"arrowstyle": "-|>", "color": INK, "lw": 1.3}
    feed_x = float(x[0]) + 0.012 * span
    ax.annotate("", xy=(feed_x, top - 0.15 * height), xytext=(feed_x, top - 0.02 * height), arrowprops=arrow)
    ax.text(feed_x + 0.012 * span, top - 0.07 * height, "sediment feed", ha="left", va="center", color=INK)
    ax.annotate("", xy=(float(x[0]) + 0.40 * span, top - 0.07 * height), xytext=(float(x[0]) + 0.30 * span, top - 0.07 * height),
                arrowprops=arrow)
    ax.text(float(x[0]) + 0.29 * span, top - 0.07 * height, "flow", ha="right", va="center", color=INK)

    water_line, = ax.plot([], [], color=WATER_LINE, lw=1.8, label="water surface (model)", zorder=4)
    bed_line, = ax.plot([], [], color=BED_LINE, lw=1.8, label="bed (model)", zorder=4)
    earlier, = ax.plot([], [], "o", color=EARLIER, ms=4, mfc="white", mew=0.9, label="earlier surveys", zorder=5)
    latest, = ax.plot([], [], "o", color=INK, ms=6, mfc="white", mew=1.2, label="measured bed, latest survey", zorder=6)
    handles = [water_line, bed_line, latest, earlier]
    # The left edge of the legend is fixed, so that it stays in place when its text changes.
    legend = {"loc": "upper left", "bbox_to_anchor": (LEGEND_LEFT, 1.0), "frameon": False, "fontsize": 12}
    ax.legend(handles=handles, **legend)
    fig.tight_layout()
    fills: list = []

    def update(k: int) -> None:
        """Draw the state after step ``k``."""
        while fills:
            fills.pop().remove()
        fills.append(ax.fill_between(x, bed[k], np.maximum(surface[k], bed[k]), color=WATER, lw=0, zorder=2))
        fills.append(ax.fill_between(x, floor, bed[k], color=DEPOSIT, lw=0, zorder=3))
        water_line.set_data(x, surface[k])
        bed_line.set_data(x, bed[k])
        done = [t for t in surveys if t <= hours[k] + 1.0e-6]
        if done:
            latest.set_data(*surveys[done[-1]])
            latest.set_label(f"measured bed, survey at {done[-1]:g} h")
        else:
            latest.set_data([], [])
        before = done[:-1]
        if before:
            earlier.set_data(np.concatenate([surveys[t][0] for t in before]), np.concatenate([surveys[t][1] for t in before]))
        else:
            earlier.set_data([], [])
        ax.legend(handles=handles, **legend)
        clock.set_text(f"time {hours[k]:5.1f} h")

    return fig, frames, update


def main() -> None:
    """Run the case (or read its history) and write the animation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="results file of learn_flume_joint.py")
    parser.add_argument("--run", type=int, choices=(1, 2, 3, 6), default=2)
    parser.add_argument("--stage", default=None, help="closure of the results file (default: the learned hiding function)")
    parser.add_argument("--minutes", type=float, default=12.0, help="time of the experiment between two frames (min)")
    parser.add_argument("--fps", type=int, default=20, help="frames per second")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--rerun", action="store_true", help="repeat the run although its history is on file")
    args = parser.parse_args()
    stages = json.loads(args.results.read_text())["stages"]
    stage = args.stage or ("hiding_network" if "hiding_network" in stages else "published")
    label = PUBLISHED_HIDING if stage == "published" and all(name in HIDING_STAGES for name in stages) else LABELS[stage]
    values = history(args.results, args.run, stage, args.rerun)
    fig, frames, update = draw(values, args.run, label, args.minutes)
    hold = round(HOLD_SECONDS * args.fps)
    base = args.results.with_name(f"{args.results.stem}_run{args.run}_{stage}")

    writers = [(base.with_suffix(".gif"), animation.PillowWriter(fps=args.fps))]
    movie = movie_writer(args.fps)
    if movie is None:
        print("ffmpeg is not on the path: no .mp4 written")
    else:
        writers.append((base.with_suffix(".mp4"), movie))
    for target, writer in writers:
        with writer.saving(fig, str(target), args.dpi):
            for k in frames:
                update(k)
                writer.grab_frame()
            for _ in range(hold):
                writer.grab_frame()
        print(f"wrote {target} ({len(frames)} frames, {len(frames) / args.fps + HOLD_SECONDS:.0f} s)")
    update(frames[-1])
    fig.savefig(base.with_suffix(".png"), dpi=200)
    print(f"wrote {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
