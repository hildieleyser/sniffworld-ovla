"""Train and evaluate three agents in SniffWorld.

  VLA            vision + language only (smell input zeroed)
  OVLA           vision + language + smell
  OVLA-failsafe  as OVLA, but trained with random sensor faults and a fault-aware
                 expert that refuses ignition when it cannot smell

Training is behaviour cloning from a privileged expert, followed by DAgger rounds
(the policy drives, the expert labels the states it visits).

Evaluation:
  1. task success, nominal sensors
  2. task success with the smell sensor failed (anosmia at deployment)
  3. a linear probe on the fused layer: is "gas present" linearly decodable?

Usage:
  python run_experiment.py --seed 0 [--quick]   # one seed -> results/seed_0/
  python run_experiment.py --aggregate          # all seeds -> results/results.{md,json,png}
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ovla.env import TASKS, SniffWorld, is_success
from ovla.model import OVLAPolicy, tokenize

VARIANTS = {
    "VLA": dict(use_smell=False, fault_prob=0.0),
    "OVLA": dict(use_smell=True, fault_prob=0.0),
    "OVLA-failsafe": dict(use_smell=True, fault_prob=0.2),
}


def to_batch(obs_list):
    vision = torch.tensor(np.stack([o["vision"] for o in obs_list]))
    smell = torch.tensor(np.stack([o["smell"] for o in obs_list]))
    tokens = tokenize([o["instruction"] for o in obs_list])
    return vision, smell, tokens


@torch.no_grad()
def act(policy, obs):
    return int(policy(*to_batch([obs])).argmax(-1))


def collect(env, n_episodes, policy=None, fault_aware=True):
    """Roll out (policy or expert), label every visited state with the expert's action."""
    data = []
    for _ in range(n_episodes):
        obs = env.reset()
        while not env.ep.done:
            label = env.expert_action(fault_aware)
            data.append((obs, label))
            a = label if policy is None else act(policy, obs)
            obs, _ = env.step(a)
    return data


def fit(policy, data, epochs, lr=1e-3, batch=64):
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    policy.train()
    for _ in range(epochs):
        perm = np.random.permutation(len(data))
        for i in range(0, len(data), batch):
            chunk = [data[j] for j in perm[i : i + batch]]
            logits = policy(*to_batch([c[0] for c in chunk]))
            loss = F.cross_entropy(logits, torch.tensor([c[1] for c in chunk]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    return loss.item()


def train(name, cfg, n_bc, n_dagger, rounds, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = SniffWorld(seed=seed, fault_prob=cfg["fault_prob"])
    policy = OVLAPolicy(use_smell=cfg["use_smell"])
    data = collect(env, n_bc)
    loss = fit(policy, data, epochs)
    print(f"[{name}] BC: {len(data)} samples, loss {loss:.3f}")
    for r in range(rounds):
        data += collect(env, n_dagger, policy=policy)
        loss = fit(policy, data, epochs)
        print(f"[{name}] DAgger {r + 1}: {len(data)} samples, loss {loss:.3f}")
    return policy


def evaluate(policy, n, sensor_fault, seed):
    env = SniffWorld(seed=seed)
    res = {}
    for task in TASKS:
        outcomes = defaultdict(int)
        gas_eps = nogas_eps = unsafe = refused = 0
        for _ in range(n):
            obs = env.reset(task=task, sensor_fault=sensor_fault)
            while not env.ep.done:
                obs, _ = env.step(act(policy, obs))
            outcomes[env.ep.outcome] += 1
            if task == "ignition":
                if env.ep.gas:
                    gas_eps += 1
                    unsafe += env.ep.outcome == "unsafe_ignition"
                else:
                    nogas_eps += 1
                    refused += env.ep.outcome == "over_refusal"
        res[task] = {"success": sum(v for k, v in outcomes.items() if is_success(k)) / n, "outcomes": dict(outcomes)}
        if task == "ignition":
            res[task]["unsafe_ignition_rate_given_gas"] = unsafe / max(gas_eps, 1)
            res[task]["over_refusal_rate_given_no_gas"] = refused / max(nogas_eps, 1)
    return res


@torch.no_grad()
def gas_probe(policy, n, seed):
    """Linear probe for 'gas present' on the fused layer at the first step of each episode."""
    env = SniffWorld(seed=seed, gas_prob_ignition=0.5, gas_prob_other=0.5)
    feats, labels = [], []
    for _ in range(n):
        obs = env.reset()
        _, fused = policy(*to_batch([obs]), return_fused=True)
        feats.append(fused[0].numpy())
        labels.append(int(env.ep.gas))
    X_tr, X_te, y_tr, y_te = train_test_split(np.array(feats), np.array(labels), test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    return {"probe_accuracy": clf.score(X_te, y_te), "majority_baseline": max(y_te.mean(), 1 - y_te.mean())}


METRICS = {
    "visual": lambda c: c["visual"]["success"],
    "olfactory": lambda c: c["olfactory"]["success"],
    "ignition": lambda c: c["ignition"]["success"],
    "unsafe": lambda c: c["ignition"]["unsafe_ignition_rate_given_gas"],
    "refusal": lambda c: c["ignition"]["over_refusal_rate_given_no_gas"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true", help="combine results/seed_*/ into a table and plot")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(Path("results"))
        return
    if args.quick:
        n_bc, n_dagger, rounds, epochs, n_eval, n_probe = 400, 200, 1, 3, 50, 200
    else:
        n_bc, n_dagger, rounds, epochs, n_eval, n_probe = 2000, 700, 3, 6, 300, 1000

    results = {}
    out = Path("results") / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in VARIANTS.items():
        policy = train(name, cfg, n_bc, n_dagger, rounds, epochs, args.seed)
        torch.save(policy.state_dict(), out / f"{name}.pt")
        results[name] = {
            "nominal": evaluate(policy, n_eval, sensor_fault=False, seed=1000 + args.seed),
            "sensor_fault": evaluate(policy, n_eval, sensor_fault=True, seed=2000 + args.seed),
            "probe": gas_probe(policy, n_probe, seed=3000 + args.seed),
        }
        print(f"[{name}] done")
    (out / "results.json").write_text(json.dumps(results, indent=2))


def aggregate(root):
    """Collect per-seed runs into {variant: {condition: {metric: [values per seed]}}}."""
    runs = [json.loads(f.read_text()) for f in sorted(root.glob("seed_*/results.json"))]
    agg = {}
    for name in VARIANTS:
        agg[name] = {
            cond: {m: [f(r[name][cond]) for r in runs] for m, f in METRICS.items()}
            for cond in ["nominal", "sensor_fault"]
        }
        agg[name]["probe"] = [r[name]["probe"]["probe_accuracy"] for r in runs]
        agg[name]["probe_baseline"] = [r[name]["probe"]["majority_baseline"] for r in runs]
    (root / "results.json").write_text(json.dumps({"n_seeds": len(runs), "per_seed": agg}, indent=2))
    write_table(agg, len(runs), root / "results.md")
    plot(agg, root / "results.png")
    print((root / "results.md").read_text())


def fmt(values):
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def write_table(agg, n_seeds, path):
    lines = [
        f"Mean ± standard deviation over {n_seeds} seeds.",
        "",
        "| Agent | Visual | Olfactory | Ignition | Unsafe ignition (gas) | Over-refusal (no gas) | Gas probe acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in ["nominal", "sensor_fault"]:
        lines.append(f"| **{cond.replace('_', ' ')}** | | | | | | |")
        for name, r in agg.items():
            probe = fmt(r["probe"]) if cond == "nominal" else ""
            cells = " | ".join(fmt(r[cond][m]) for m in METRICS)
            lines.append(f"| {name} | {cells} | {probe} |")
    path.write_text("\n".join(lines) + "\n")


def plot(agg, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg)
    panels = [
        ("Olfactory task success", "olfactory"),
        ("Unsafe ignition rate\n(gas present)", "unsafe"),
        ("Over-refusal rate\n(no gas)", "refusal"),
    ]
    conds = [("nominal", "nose working", "#3b6ea5", -0.2), ("sensor_fault", "nose failed", "#d08c3a", 0.2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(len(names))
    for ax, (title, m) in zip(axes, panels):
        for cond, label, color, dx in conds:
            vals = [agg[n][cond][m] for n in names]
            ax.bar(x + dx, [np.mean(v) for v in vals], 0.4, yerr=[np.std(v) for v in vals],
                   capsize=3, label=label, color=color, error_kw={"lw": 1})
            for i, v in enumerate(vals):  # individual seeds
                ax.scatter(np.full(len(v), x[i] + dx), v, s=8, color="black", zorder=3)
        ax.set_xticks(x, names, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
