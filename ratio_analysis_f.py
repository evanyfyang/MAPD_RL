import matplotlib.pyplot as plt

f_values = [0.2, 0.5, 1, 2, 5, 10]
f_positions = list(range(len(f_values)))

agent_values = [10, 20, 30, 40, 50]
agent_positions = list(range(len(agent_values)))

# STAR-GNN < Hungarian (f)
ratio_STAR_GNN_lt_f = [5.12, 9.48, 14.36, 19.82, 24.97, 26.88]

# STAR-GNN < Hungarian (agent number)
ratio_STAR_GNN_lt_agent = [11.54, 15.92, 20.63, 23.18, 24.97]

plt.figure(figsize=(6.5, 4.2))
ax = plt.gca()
ax.plot(
    f_positions,
    ratio_STAR_GNN_lt_f,
    marker="o",
    color="tab:orange",
    label="STAR-GNN < Hungarian (f)",
)

ax.set_xlabel("f")
ax.set_ylabel("ratio (%)")
ax.set_xticks(f_positions, f_values)
plt.yticks(range(0, 101, 2))

ax_top = ax.twiny()
ax_top.set_xlim(min(agent_positions) - 0.5, max(agent_positions) + 0.5)
ax_top.plot(
    agent_positions,
    ratio_STAR_GNN_lt_agent,
    marker="o",
    color="tab:blue",
    label="STAR-GNN < Hungarian (agent number)",
)
ax_top.set_xlabel("agent number")
ax_top.set_xticks(agent_positions, agent_values)

handles_bottom, labels_bottom = ax.get_legend_handles_labels()
handles_top, labels_top = ax_top.get_legend_handles_labels()
ax.legend(handles_bottom + handles_top, labels_bottom + labels_top, loc="best")
plt.tight_layout()
plt.savefig("ratio_plot_combined.png", dpi=300)
