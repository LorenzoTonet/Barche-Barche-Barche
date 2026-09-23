import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse

from source_code.Environment.environment import SailingEnv, FlattenSailingObs
from source_code.Environment.environment_generators import create_random_environment
from source_code.Environment.map_elements import Checkpoint
from source_code.Agent.PPO import PPOAgent


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
parser.add_argument('--agent', type=str, default='./checkpoints/ppo_sailing.pt', help="Path to agent model to load")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

agent.load(args.agent)

cfg["max_steps"] = 1500
cfg["train"]["env"]["variable_n_checkpoints"] = False

checkpoint_counts = [1, 2, 3, 4]

n_tests_per_config = 200

results = {}

print("=== STARTING BENCHMARK ===")

for n_cp in checkpoint_counts:
    cfg["train"]["env"]["n_checkpoints"] = n_cp
    
    rewards_list = []
    steps_list = []
    successes = 0
    truncated_runs = 0
    oob = 0
    
    for i in range(n_tests_per_config):
        env = create_random_environment(cfg, verbose = False)
        env = FlattenSailingObs(env)

        state, _ = env.reset()
        cum_reward = 0
        steps = 0
        done = truncated = False
        
        while not (done or truncated):
            action, _ = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            cum_reward += reward
            steps += 1
        
        rewards_list.append(cum_reward)
        steps_list.append(steps)
        
        if done and not info["failed"]:
            successes += 1

        if info["failed"]:
            oob += 1

        if truncated: truncated_runs += 1
            
    # Stats
    avg_reward = np.mean(rewards_list)
    std_reward = np.std(rewards_list)
    avg_steps = np.mean(steps_list)
    success_rate = (successes / n_tests_per_config) * 100
    truncated_rate = (truncated_runs / n_tests_per_config) * 100
    oob_rate = (oob / n_tests_per_config) * 100

    results[n_cp] = {
        "rewards": rewards_list,
        "steps": steps_list,
        "mean_reward": avg_reward,
        "std_reward": std_reward,
        "mean_steps": avg_steps,
        "success_rate": success_rate,
        "truncated_rate": truncated_rate,
        "out_of_border_rate": oob_rate
    }
    
    print(f"Checkpoints: {n_cp:2d} | Avg. Reward: {avg_reward:8.2f} ± {std_reward:6.2f} | Avg. Steps: {avg_steps:6.1f} | Success rate: {success_rate:5.1f}%")

print("=== BENCHMARK COMPLETED ===\n")

# PLOTS (By claude)
# Assicuriamoci che i checkpoint siano ordinati
checkpoint_counts = sorted(checkpoint_counts)

# ------------------------------------------------------------
# Estrazione dati
# ------------------------------------------------------------

rewards_data = [results[n]["rewards"] for n in checkpoint_counts]

means = np.array([
    results[n]["mean_reward"]
    for n in checkpoint_counts
])

stds = np.array([
    results[n]["std_reward"]
    for n in checkpoint_counts
])

steps_data = np.array([
    results[n]["mean_steps"]
    for n in checkpoint_counts
])

success_data = np.array([
    results[n]["success_rate"]
    for n in checkpoint_counts
])

truncated_data = np.array([
    results[n]["truncated_rate"]
    for n in checkpoint_counts
])

oob_data = np.array([
    results[n]["out_of_border_rate"]
    for n in checkpoint_counts
])

positions = np.arange(len(checkpoint_counts))


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


# ============================================================
# 1. REWARD DISTRIBUTION
# ============================================================

ax = axes[0, 0]

# Violin plot
parts = ax.violinplot(
    rewards_data,
    positions=positions,
    showmeans=False,
    showmedians=True,
    showextrema=True
)

# Individual samples con jitter
for i, rewards in enumerate(rewards_data):

    rewards = np.asarray(rewards)

    jitter = np.random.normal(
        loc=positions[i],
        scale=0.055,
        size=len(rewards)
    )

    ax.scatter(
        jitter,
        rewards,
        s=12,
        alpha=0.20
    )

# Mean ± std
ax.errorbar(
    positions,
    means,
    yerr=stds,
    fmt='o-',
    linewidth=2,
    markersize=6,
    capsize=4,
    label='Mean ± std'
)

ax.set_xticks(positions)
ax.set_xticklabels(checkpoint_counts)

ax.set_title(
    "Reward Distribution",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Cumulative Reward")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend()
# ============================================================
# 2. EPISODE OUTCOME DISTRIBUTION
# ============================================================

ax = axes[0, 1]

# Matrice:
# colonne = [success, truncated, out-of-border]
outcomes = np.vstack([
    success_data,
    truncated_data,
    oob_data
]).T

ax.bar(
    checkpoint_counts,
    outcomes[:, 0],
    label="Success"
)

ax.bar(
    checkpoint_counts,
    outcomes[:, 1],
    bottom=outcomes[:, 0],
    label="Truncated"
)

ax.bar(
    checkpoint_counts,
    outcomes[:, 2],
    bottom=outcomes[:, 0] + outcomes[:, 1],
    label="Out of border"
)

# ------------------------------------------------------------
# Percentuali dentro le barre
# ------------------------------------------------------------

for i, n_cp in enumerate(checkpoint_counts):

    cumulative = 0

    values = [
        success_data[i],
        truncated_data[i],
        oob_data[i]
    ]

    for value in values:

        if value >= 5:

            ax.text(
                n_cp,
                cumulative + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold"
            )

        cumulative += value


# ------------------------------------------------------------
# Controllo: gli outcome devono sommare a 100%
# ------------------------------------------------------------

outcome_sums = (
    success_data
    + truncated_data
    + oob_data
)

if not np.allclose(outcome_sums, 100.0):
    print(
        "WARNING: episode outcomes do not sum to 100%:",
        outcome_sums
    )


# ------------------------------------------------------------
# Formattazione grafico
# ------------------------------------------------------------

ax.set_title(
    "Episode Outcome Distribution",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Episodes (%)")

ax.set_ylim(0, 100)
ax.set_xticks(checkpoint_counts)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend()
# ============================================================
# 3. AVERAGE EPISODE LENGTH
# ============================================================

ax = axes[0, 2]

ax.plot(
    checkpoint_counts,
    steps_data,
    marker='s',
    linewidth=2.5,
    markersize=7
)

ax.set_title(
    "Average Episode Duration",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Average Steps")

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

# Annotazioni valori
for x, y in zip(checkpoint_counts, steps_data):
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# ============================================================
# 4. SUCCESS RATE
# ============================================================

ax = axes[1, 0]

ax.plot(
    checkpoint_counts,
    success_data,
    marker='o',
    linewidth=2.5,
    markersize=7
)

ax.fill_between(
    checkpoint_counts,
    success_data,
    alpha=0.12
)

ax.set_title(
    "Success Rate",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Success Rate (%)")

ax.set_ylim(0, 105)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

for x, y in zip(checkpoint_counts, success_data):

    ax.annotate(
        f"{y:.1f}%",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# ============================================================
# 5. TRUNCATED RATE
# ============================================================

ax = axes[1, 1]

ax.plot(
    checkpoint_counts,
    truncated_data,
    marker='s',
    linewidth=2.5,
    markersize=7
)

ax.fill_between(
    checkpoint_counts,
    truncated_data,
    alpha=0.12
)

ax.set_title(
    "Truncated Episode Rate",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Truncated Rate (%)")

ax.set_ylim(0, 105)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

for x, y in zip(checkpoint_counts, truncated_data):

    ax.annotate(
        f"{y:.1f}%",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# ============================================================
# 6. OUT-OF-BORDER RATE
# ============================================================

ax = axes[1, 2]

ax.plot(
    checkpoint_counts,
    oob_data,
    marker='^',
    linewidth=2.5,
    markersize=7
)

ax.fill_between(
    checkpoint_counts,
    oob_data,
    alpha=0.12
)

ax.set_title(
    "Out-of-Border Rate",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Number of Checkpoints")
ax.set_ylabel("Out-of-Border Rate (%)")

ax.set_ylim(0, 105)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

for x, y in zip(checkpoint_counts, oob_data):

    ax.annotate(
        f"{y:.1f}%",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# ============================================================
# GENERAL TITLE
# ============================================================

fig.suptitle(
    "Benchmark Model Performance",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()