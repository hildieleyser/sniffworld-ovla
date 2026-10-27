"""Render the README figures from results/ and the seed-0 checkpoints.

Every figure is drawn twice, for GitHub's light and dark themes:
  figures/<name>-light.png, figures/<name>-dark.png

Usage:  python make_figures.py      (after run_experiment.py and --aggregate)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ovla.env import CONTAINER_RGB, GRID, HEATER_RGB, ODORS, SniffWorld
from ovla.model import OVLAPolicy
from run_experiment import VARIANTS, act, to_batch

OUT = Path("figures")
AGENTS = list(VARIANTS)

# Validated palette (dataviz reference instance): categorical slots 1-3, chrome and ink.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        agents=["#2a78d6", "#eb6834", "#1baf7a"], critical="#d03b3b", neutral="#c3c2b7",
        seq=["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        seq_gas=["#fcfcfb", "#fbe0d3", "#f4b293", "#eb6834", "#b8461b", "#7a2a0c"],
        field=["#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781"],
    ),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        agents=["#3987e5", "#d95926", "#199e70"], critical="#d03b3b", neutral="#5f5e5a",
        seq=["#1a1a19", "#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
        seq_gas=["#1a1a19", "#5a210b", "#9a3a15", "#d95926", "#f09a74", "#fbe0d3"],
        field=["#1a1a19", "#2c2c2a", "#383835", "#5f5e5a"],
    ),
}
FONT = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def style(t):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 9,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": t["ink"], "axes.titlelocation": "left", "axes.grid": False,
        "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "axes.linewidth": 0.8,
    })


def save(fig, name, theme):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def cmap(colors, name):
    return LinearSegmentedColormap.from_list(name, colors)


def load(name, seed=0):
    p = OVLAPolicy(use_smell=VARIANTS[name]["use_smell"])
    p.load_state_dict(torch.load(f"results/seed_{seed}/{name}.pt"))
    return p.eval()


def rollout(policy, env_seed, **reset_kw):
    env = SniffWorld(seed=env_seed)
    obs = env.reset(**reset_kw)
    path, last = [env.ep.agent], None
    while not env.ep.done:
        last = act(policy, obs)
        obs, _ = env.step(last)
        path.append(env.ep.agent)
    return env, np.array(path), last


def draw_room(ax, env, t, labels=True, alpha=1.0):
    """Objects as rounded squares; text labels instead of colour alone."""
    names = {CONTAINER_RGB: "box", HEATER_RGB: "heater"}
    for (x, y), rgb in env.ep.objects:
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - 0.3, y - 0.3), 0.6, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
            fc=rgb, ec=t["surface"], lw=1.5, zorder=3, alpha=alpha))
        if labels and tuple(rgb) in names:
            ax.text(x, y + 0.55, names[tuple(rgb)], ha="center", va="top", fontsize=6.5, color=t["ink2"], zorder=4)


def grid_axes(ax, t):
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(t["axis"])


# --------------------------------------------------------------------------- 1. the world


def fig_world(t, theme):
    env = SniffWorld(seed=11)
    env.reset(task="olfactory", gas=True)
    field = env._field
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), gridspec_kw={"wspace": 0.08})
    ax = axes[0]
    ax.imshow(np.full((GRID, GRID, 3), matplotlib.colors.to_rgb(t["surface"])))
    for k in range(GRID + 1):
        ax.axhline(k - 0.5, color=t["grid"], lw=0.6)
        ax.axvline(k - 0.5, color=t["grid"], lw=0.6)
    draw_room(ax, env, t)
    ax.scatter(*env.ep.agent, s=90, marker="o", color=t["ink"], zorder=5)
    ax.text(env.ep.agent[0], env.ep.agent[1] + 0.5, "agent", ha="center", va="top", fontsize=6.5,
            color=t["ink"], fontweight="bold")
    grid_axes(ax, t)
    ax.set_title("What the camera sees", fontsize=9)
    ax.text(0, -0.04, "Gas is invisible and the three\nboxes look identical.", transform=ax.transAxes,
            va="top", fontsize=7.5, color=t["ink2"])

    log_f = np.log10(field + 1e-5)
    for i, o in enumerate(ODORS):
        ax = axes[i + 1]
        cm = cmap(t["seq_gas"] if o == "gas" else t["seq"], o)
        im = ax.imshow(log_f[i], cmap=cm, vmin=-3, vmax=0, interpolation="bicubic")
        ax.contour(log_f[i], levels=[-2.5, -2, -1.5, -1, -0.5], colors=t["surface"], linewidths=0.5, alpha=0.6, linestyles="solid")
        draw_room(ax, env, t, labels=False, alpha=0.35)
        (sx, sy), *_ = [src for src in env.ep.odor_sources if src[1] == i][0]
        ax.add_patch(matplotlib.patches.Circle((sx, sy), 0.55, fc="none", ec=t["ink"], lw=1.5, zorder=5))
        ax.text(sx, sy + 0.75 if sy < GRID - 2 else sy - 0.75, "leak" if o == "gas" else "source",
                ha="center", va="top" if sy < GRID - 2 else "bottom", fontsize=7, color=t["ink"],
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=t["surface"], ec="none", alpha=0.8))
        grid_axes(ax, t)
        ax.set_title(f"{o}", fontsize=9)
        ax.text(0, -0.04, "broad: smelled everywhere" if o == "gas" else "narrow: points to one box",
                transform=ax.transAxes, va="top", fontsize=7.5, color=t["ink2"])
    cb = fig.colorbar(im, ax=axes[1:], fraction=0.015, pad=0.01)
    cb.set_label("log₁₀ concentration", color=t["ink2"], fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelsize=7)
    fig.suptitle("SniffWorld: one room, four odour plumes", x=0.125, y=1.1, ha="left", fontsize=12,
                 fontweight="bold", color=t["ink"])
    fig.text(0.125, 0.99, "Each container emits one odour; a gas leak spreads a broad invisible plume. "
             "The agent samples all four at its cell and four neighbours.", fontsize=8.5, color=t["ink2"])
    save(fig, "1-world", theme)


# --------------------------------------------------------------------------- 2. trajectories


def find_episode(task, gas, want, start=0):
    """First env seed where each agent's outcome is in want[agent]."""
    policies = {n: load(n) for n in want}
    for s in range(start, start + 500):
        outs = {n: rollout(p, s, task=task, gas=gas, sensor_fault=f)
                for n, (p, f) in ((n, (policies[n], want[n][1])) for n in want)}
        if all(outs[n][0].ep.outcome in want[n][0] for n in want):
            return s, outs
    raise RuntimeError("no matching episode")


def fig_trajectories(t, theme):
    color = dict(zip(AGENTS, t["agents"]))
    rows = [
        ("olfactory", False, {"VLA": ({"wrong_target", "timeout"}, False), "OVLA": ({"success"}, False)},
         "“find the box with the {} smell”"),
        ("ignition", True, {"VLA": ({"unsafe_ignition"}, False), "OVLA": ({"correct_halt"}, False)},
         "“light the heater” (gas leak present)"),
        ("ignition", True, {"OVLA": ({"unsafe_ignition"}, True), "OVLA-failsafe": ({"correct_halt"}, True)},
         "“light the heater” (gas leak, nose broken)"),
    ]
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 10.6), gridspec_kw={"hspace": 0.38, "wspace": 0.06})
    for r, (task, gas, want, caption) in enumerate(rows):
        seed, outs = find_episode(task, gas, want, start=40 * r)
        for c, name in enumerate(want):
            env, path, last = outs[name]
            ax = axes[r, c]
            if task == "olfactory":
                target_odor = [o for o in ODORS if o in env.ep.instruction][0]
                bg = np.log10(env._field[ODORS.index(target_odor)] + 1e-5)
                cap = caption.format(target_odor)
            else:
                bg = np.log10(env._field[ODORS.index("gas")] + 1e-5)
                cap = caption
            ax.imshow(bg, cmap=cmap(t["field"], "f"), vmin=-3, vmax=0, interpolation="bicubic")
            draw_room(ax, env, t, labels=False, alpha=0.45)
            tx, ty = env.ep.target
            ax.add_patch(matplotlib.patches.Circle((tx, ty), 0.55, fc="none", ec=t["ink"], lw=1.5, zorder=5))
            above = ty > 0
            ax.text(tx, ty - 0.62 if above else ty + 0.62, "heater" if task == "ignition" else "target",
                    ha="center", va="bottom" if above else "top",
                    fontsize=7, color=t["ink"], fontweight="bold", zorder=9)
            ax.plot(path[:, 0], path[:, 1], color=color[name], lw=2.5, zorder=6,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.scatter(*path[0], s=70, color=t["surface"], edgecolor=color[name], lw=2.5, zorder=7)
            outcome = env.ep.outcome
            label = {"correct_halt": "halted", "success": "success", "unsafe_ignition": "ignited in gas",
                     "wrong_target": "wrong box", "timeout": "timed out"}[outcome]
            if outcome == "correct_halt":
                hx = path[-1][0]
                ax.text(hx + 0.55 if hx < GRID - 2 else hx - 0.55, path[-1][1], "halt", va="center",
                        ha="left" if hx < GRID - 2 else "right", fontsize=8, fontweight="bold",
                        color=t["ink"], zorder=9, bbox=dict(boxstyle="round,pad=0.2", fc=t["surface"],
                                                             ec=color[name], lw=1.5))
            elif outcome != "success":
                ax.scatter(*path[-1], s=140, marker="X", color=t["critical"], edgecolor=t["surface"],
                           lw=1.5, zorder=8)
            else:
                ax.scatter(*path[-1], s=90, color=color[name], edgecolor=t["surface"], lw=1.5, zorder=8)
            grid_axes(ax, t)
            sensor = " · nose broken" if want[name][1] else ""
            ax.set_title(f"{name}{sensor}", fontsize=9.5)
            ax.add_line(Line2D([0, 0.06], [1.035, 1.035], transform=ax.transAxes, color=color[name], lw=3,
                               clip_on=False))
            ax.title.set_position((0.08, 1.0))
            steps = len(path) - 1
            ax.text(0.0, -0.04, f"{label} after {steps} step{'s' * (steps != 1)}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    color=t["critical"] if outcome in ("unsafe_ignition",) else t["ink2"],
                    fontweight="bold" if outcome == "unsafe_ignition" else "normal")
            if c == 0:
                ax.annotate(cap, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                            fontsize=9.5, color=t["ink"], style="italic", va="bottom")
    fig.text(0.125, 0.985, "Same room, same instruction, different senses", ha="left", va="bottom",
             fontsize=12, fontweight="bold")
    fig.text(0.125, 0.978, "Shading: the plume that matters (target odour, or gas). Hollow dot = start; "
             "line = path; red X = failure.", fontsize=8.5, color=t["ink2"], va="top")
    save(fig, "2-trajectories", theme)


# --------------------------------------------------------------------------- 3. main results


def fig_results(t, theme, agg):
    color = dict(zip(AGENTS, t["agents"]))
    panels = [
        ("Olfactory task success", "olfactory", "higher is better"),
        ("Ignites heater despite gas", "unsafe", "lower is better"),
        ("Refuses a safe ignition", "refusal", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, gridspec_kw={"wspace": 0.12})
    ys = {n: i for i, n in enumerate(reversed(AGENTS))}
    for ax, (title, m, note) in zip(axes, panels):
        for x in (0, 0.25, 0.5, 0.75, 1):
            ax.axvline(x, color=t["grid"], lw=0.8, zorder=0)
        for n in AGENTS:
            for cond, dy, filled in (("nominal", 0.14, True), ("sensor_fault", -0.14, False)):
                v = np.array(agg[n][cond][m])
                y = ys[n] + dy
                ax.plot([v.mean() - v.std(), v.mean() + v.std()], [y, y], color=color[n], lw=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter(v, np.full(len(v), y), s=10, color=t["ink2"], zorder=3, lw=0)
                ax.scatter(v.mean(), y, s=70, color=color[n] if filled else t["surface"],
                           edgecolor=color[n], lw=2, zorder=4)
                right = max(v.max(), v.mean() + v.std()) + 0.04
                left = min(v.min(), v.mean() - v.std()) - 0.04
                ax.text(right if right < 0.9 else left, y, f"{v.mean():.2f}", va="center",
                        ha="left" if right < 0.9 else "right", fontsize=7.5, color=t["ink2"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(title, pad=16)
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=7.5, color=t["muted"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(ys.values()), list(ys))
    for lbl in axes[0].get_yticklabels():
        lbl.set_color(color[lbl.get_text()])
        lbl.set_fontweight("bold")
    handles = [
        Line2D([], [], marker="o", ls="", ms=7, color=t["ink2"], label="nose working"),
        Line2D([], [], marker="o", ls="", ms=7, mfc=t["surface"], mec=t["ink2"], mew=2, label="nose broken"),
        Line2D([], [], marker="o", ls="", ms=3, color=t["ink2"], label="individual seed"),
        Line2D([], [], color=t["ink2"], lw=2, label="±1 SD"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=4, bbox_to_anchor=(0.12, -0.08), fontsize=8)
    fig.suptitle("A nose fixes the chemical tasks, and a broken nose is silently catastrophic",
                 x=0.125, y=1.13, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.125, 1.02, f"Mean over {len(agg['VLA']['probe'])} training seeds, 300 test episodes per task "
             "and seed, on unseen layouts.", fontsize=8.5, color=t["ink2"])
    save(fig, "3-results", theme)


# --------------------------------------------------------------------------- 4. safety trade-off plane


def fig_tradeoff(t, theme, agg):
    color = dict(zip(AGENTS, t["agents"]))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.fill_between([0, 1], [1, 0], [1, 1], color=t["grid"], alpha=0.45, lw=0, zorder=0)
    ax.plot([0, 1], [1, 0], color=t["muted"], lw=1, zorder=1)
    ax.text(0.56, 0.36, "no-information line", rotation=-45, rotation_mode="anchor", fontsize=7.5,
            color=t["muted"], ha="center", va="center")
    ax.text(0.97, 0.97, "worse than\nguessing", fontsize=7.5, color=t["muted"], ha="right", va="top")
    ax.annotate("ideal corner: OVLA and OVLA-failsafe\nwith a working nose, and OVLA-failsafe\n"
                "with a broken nose", xy=(0.015, 0.015), xytext=(0.12, 0.13), fontsize=7.5, color=t["ink2"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8, shrinkA=2, shrinkB=6))
    for n in AGENTS:
        x = np.array(agg[n]["nominal"]["refusal"])
        conds = [("nominal", True)] + ([("sensor_fault", False)] if n != "VLA" else [])
        for cond, filled in conds:
            ax.scatter(x, agg[n][cond]["unsafe"], s=75, color=color[n] if filled else t["surface"],
                       edgecolor=color[n], lw=2.2, zorder=4, clip_on=False)
    ax.text(0.07, 0.985, "OVLA, nose broken:\nignites into gas", fontsize=8, color=t["ink"],
            fontweight="bold", va="top")
    ax.text(0.47, 0.72, "VLA (no nose): each seed\nsettles on a different habit", fontsize=8, color=t["ink"],
            fontweight="bold")
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Over-refusal rate (refuses when there is no gas)")
    ax.set_ylabel("Unsafe ignition rate (ignites when there is gas)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=n) for n, c in color.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=7, mfc=t["surface"], mec=t["ink2"], mew=2,
                       label="same agent,\nnose broken")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, labelspacing=0.9)
    fig.suptitle("The safety trade-off plane", x=0.125, y=1.0, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.125, 0.955, "One point per trained agent (3 seeds each). On the diagonal, refusals and unsafe\n"
             "ignitions sum to 1: the agent acts on a fixed habit, not on what it senses.\n"
             "Hollow points keep the agent's working-nose over-refusal on the x-axis.",
             fontsize=8, color=t["ink2"], va="top")
    fig.subplots_adjust(top=0.82)
    save(fig, "4-tradeoff", theme)


# --------------------------------------------------------------------------- 5. internal gas representation


@torch.no_grad()
def fused_states(policy, n=1000, seed=3000):
    env = SniffWorld(seed=seed, gas_prob_ignition=0.5, gas_prob_other=0.5)
    feats, gas = [], []
    for _ in range(n):
        obs = env.reset()
        feats.append(policy(*to_batch([obs]), return_fused=True)[1][0].numpy())
        gas.append(env.ep.gas)
    return np.array(feats), np.array(gas)


def fig_probe(t, theme, agg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    color = dict(zip(AGENTS, t["agents"]))
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.28)
    bins = np.linspace(0, 1, 26)
    for i, n in enumerate(AGENTS):
        ax = fig.add_subplot(gs[i])
        X, g = fused_states(load(n))
        X_tr, X_te, g_tr, g_te = train_test_split(X, g, test_size=0.3, random_state=0)
        clf = LogisticRegression(max_iter=2000).fit(X_tr, g_tr)
        z = clf.predict_proba(X_te)[:, 1]
        # Gas presence is a hazard *state*: status-critical with a label, against neutral.
        ax.hist(z[~g_te], bins=bins, color=t["neutral"], label="no gas", rwidth=0.85)
        ax.hist(z[g_te], bins=bins, color=t["critical"], label="gas present", rwidth=0.85, alpha=0.85)
        ax.axvline(0.5, color=t["muted"], lw=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(n, pad=12)
        ax.add_line(Line2D([0, 0.08], [1.03, 1.03], transform=ax.transAxes, color=color[n], lw=3,
                           clip_on=False))
        ax.set_xlabel("probe P(gas), held-out episodes", fontsize=7.5)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.text(0.5, 1.0, f"acc. {clf.score(X_te, g_te):.2f}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color=t["ink2"])
        if i == 0:
            ax.legend(loc="upper left", fontsize=7.5, handlelength=1, handletextpad=0.4)
    ax = fig.add_subplot(gs[3])
    base = np.mean([np.mean(agg[n]["probe_baseline"]) for n in AGENTS])
    ax.axhline(base, color=t["muted"], lw=1)
    ax.text(0, 1.015, f"line: majority-class baseline ({base:.2f})", transform=ax.transAxes, fontsize=7.5,
            color=t["muted"])
    for i, n in enumerate(AGENTS):
        v = np.array(agg[n]["probe"])
        ax.bar(i, v.mean(), 0.55, color=color[n], zorder=2)
        ax.scatter(np.full(len(v), i), v, s=14, color=t["ink"], zorder=3, lw=0)
        ax.text(i, max(v.max(), v.mean()) + 0.025, f"{v.mean():.2f}", ha="center", fontsize=8, color=t["ink2"])
    ax.set_xticks(range(3), AGENTS, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Probe accuracy, 3 seeds", pad=12)
    fig.suptitle("The hazard is written into the agent's internals, but only if it has a nose",
                 x=0.125, y=1.08, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.125, 0.995, "A linear probe reads “gas present” from the fused layer at the first step. "
             "Left: probe scores on held-out episodes (seed 0); separated humps mean the hazard is "
             "linearly readable.", fontsize=8.5, color=t["ink2"])
    save(fig, "5-probe", theme)


def main():
    agg = json.loads(Path("results/results.json").read_text())["per_seed"]
    for theme, t in THEMES.items():
        style(t)
        fig_world(t, theme)
        fig_trajectories(t, theme)
        fig_results(t, theme, agg)
        fig_tradeoff(t, theme, agg)
        fig_probe(t, theme, agg)
        print(f"rendered {theme}")


if __name__ == "__main__":
    main()
