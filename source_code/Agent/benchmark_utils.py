import copy
import os

import numpy as np
import matplotlib.pyplot as plt


from source_code.Environment.environment import SailingEnv, FlattenSailingObs
from source_code.Environment.environment_generators import create_random_environment
from source_code.Environment.map_elements import Checkpoint
from source_code.Agent.PPO import PPOAgent

def run_benchmark(agent, cfg, checkpoint_counts=(1, 2, 3, 4), n_tests=200,
                  max_steps=1500, seed=None, verbose=True):
    cfg = copy.deepcopy(cfg)  # niente side effect sul cfg del chiamante
    cfg["max_steps"] = max_steps
    cfg["train"]["env"]["variable_n_checkpoints"] = False

    results = {}
    for n_cp in checkpoint_counts:
        cfg["train"]["env"]["n_checkpoints"] = n_cp
        rewards, steps = [], []
        counts = {"success": 0, "truncated": 0, "oob": 0}

        for _ in range(n_tests):
            env = FlattenSailingObs(create_random_environment(cfg, verbose=False))
            state, _ = env.reset()
            cum_reward, n_steps = 0.0, 0
            done = truncated = False
            info = {}

            while not (done or truncated):
                action, _ = agent.get_action(state, deterministic=True)
                state, r, done, truncated, info = env.step(action)
                cum_reward += r
                n_steps += 1
            env.close()

            rewards.append(cum_reward)
            steps.append(n_steps)

            # classificazione mutuamente esclusiva -> somma sempre 100%
            if info.get("failed", False):
                counts["oob"] += 1
            elif done:
                counts["success"] += 1
            else:
                counts["truncated"] += 1

        results[n_cp] = {
            "rewards": rewards,
            "steps": steps,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "success_rate": 100 * counts["success"] / n_tests,
            "truncated_rate": 100 * counts["truncated"] / n_tests,
            "out_of_border_rate": 100 * counts["oob"] / n_tests,
        }
        if verbose:
            r = results[n_cp]
            print(f"Checkpoints: {n_cp:2d} | Avg. Reward: {r['mean_reward']:8.2f} "
                  f"± {r['std_reward']:6.2f} | Avg. Steps: {r['mean_steps']:6.1f} "
                  f"| Success: {r['success_rate']:5.1f}%")
    return results


def plot_benchmark(results, save_path, title="Benchmark Model Performance",
                   seed=0, dpi=200):
    cps = sorted(results)
    pos = np.arange(len(cps))
    rng = np.random.default_rng(seed)

    get = lambda k: np.array([results[n][k] for n in cps])
    means, stds = get("mean_reward"), get("std_reward")
    steps, succ = get("mean_steps"), get("success_rate")
    trunc, oob = get("truncated_rate"), get("out_of_border_rate")
    rewards_data = [results[n]["rewards"] for n in cps]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def style(ax, t, ylabel, ylim=None):
        ax.set_title(t, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Checkpoints")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.3)

    # 1. Reward distribution
    ax = axes[0, 0]
    ax.violinplot(rewards_data, positions=pos, showmeans=False,
                  showmedians=True, showextrema=True)
    for i, r in enumerate(rewards_data):
        ax.scatter(rng.normal(pos[i], 0.055, len(r)), r, s=12, alpha=0.2)
    ax.errorbar(pos, means, yerr=stds, fmt="o-", lw=2, ms=6, capsize=4,
                label="Mean ± std")
    ax.set_xticks(pos)
    ax.set_xticklabels(cps)
    style(ax, "Reward Distribution", "Cumulative Reward")
    ax.legend()

    # 2. Episode outcomes (stacked)
    ax = axes[0, 1]
    bottom = np.zeros(len(cps))
    for vals, lab in [(succ, "Success"), (trunc, "Truncated"), (oob, "Out of border")]:
        ax.bar(cps, vals, bottom=bottom, label=lab)
        for x, v, b in zip(cps, vals, bottom):
            if v >= 5:
                ax.text(x, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        bottom += vals
    ax.set_xticks(cps)
    style(ax, "Episode Outcome Distribution", "Episodes (%)", (0, 100))
    ax.legend()

    # 3-6. Line plots con annotazioni
    def line(ax, y, marker, t, ylabel, fmt, ylim=None, fill=False):
        ax.plot(cps, y, marker=marker, lw=2.5, ms=7)
        if fill:
            ax.fill_between(cps, y, alpha=0.12)
        for x, v in zip(cps, y):
            ax.annotate(fmt.format(v), (x, v), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(cps)
        style(ax, t, ylabel, ylim)

    line(axes[0, 2], steps, "s", "Average Episode Duration", "Average Steps", "{:.1f}")
    line(axes[1, 0], succ, "o", "Success Rate", "Success Rate (%)", "{:.1f}%", (0, 105), True)
    line(axes[1, 1], trunc, "s", "Truncated Episode Rate", "Truncated Rate (%)", "{:.1f}%", (0, 105), True)
    line(axes[1, 2], oob, "^", "Out-of-Border Rate", "Out-of-Border Rate (%)", "{:.1f}%", (0, 105), True)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def benchmark_agent(agent, cfg, save_dir="./benchmarks", filename="benchmark.png",
                    checkpoint_counts=(1, 2, 3, 4), n_tests=200, max_steps=1500,
                    seed=None, title="Benchmark Model Performance", verbose=True):
    if seed is not None:
        np.random.seed(seed)
        # se create_random_environment usa altri RNG (torch, random, ...) seedali qui

    if verbose:
        print("=== STARTING BENCHMARK ===")
    results = run_benchmark(agent, cfg, checkpoint_counts, n_tests, max_steps, seed, verbose)
    path = plot_benchmark(results, os.path.join(save_dir, filename), title=title)
    if verbose:
        print(f"=== BENCHMARK COMPLETED === plot salvato in {path}\n")
    return results