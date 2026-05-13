import matplotlib.pyplot as plt

f_values = [0.2, 0.5, 1, 2, 5, 10]
f_positions = list(range(len(f_values)))

agent_values = [10, 20, 30, 40, 50]
agent_positions = list(range(len(agent_values)))

# STAR-GNN < Hungarian (f)
ratio_STAR_GNN_lt_f = [5.12, 9.48, 14.36, 19.82, 24.97, 26.88]

# STAR-GNN < Hungarian (agent number)
ratio_STAR_GNN_lt_agent = [11.54, 15.92, 20.63, 23.18, 24.97]

# Paper-friendly style: larger fonts, compact canvas.
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
    }
)

plt.figure(figsize=(4.8, 3.1))
ax = plt.gca()
ax.plot(
    f_positions,
    ratio_STAR_GNN_lt_f,
    marker="o",
    color="tab:orange",
    linewidth=2.0,
    markersize=5,
    label="STAR-GNN < Hungarian (f)",
)

ax.set_xlabel("f")
ax.set_ylabel("ratio (%)")
ax.set_xticks(f_positions, f_values)
ax.set_ylim(0, 30)
ax.set_yticks(range(0, 31, 5))
ax.grid(axis="y", linestyle="--", alpha=0.3)

ax_top = ax.twiny()
ax_top.set_xlim(min(agent_positions) - 0.5, max(agent_positions) + 0.5)
ax_top.plot(
    agent_positions,
    ratio_STAR_GNN_lt_agent,
    marker="o",
    color="tab:blue",
    linewidth=2.0,
    markersize=5,
    label="STAR-GNN < Hungarian (agent number)",
)
ax_top.set_xlabel("agent number")
ax_top.set_xticks(agent_positions, agent_values)

handles_bottom, labels_bottom = ax.get_legend_handles_labels()
handles_top, labels_top = ax_top.get_legend_handles_labels()
ax.legend(handles_bottom + handles_top, labels_bottom + labels_top, loc="lower right", frameon=False)
plt.tight_layout(pad=0.2)
plt.savefig("ratio_plot_combined.png", dpi=300, bbox_inches="tight")
