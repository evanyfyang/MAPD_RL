import matplotlib.pyplot as plt

agents = [10, 20, 30, 40, 50]

# TAPF < Hungarian
ratio_tapf_lt = [11.54, 15.92, 20.63, 23.18, 24.97]

plt.figure(figsize=(6.5, 4.2))
ax = plt.gca()
ax.plot(agents, ratio_tapf_lt, marker="o", color="tab:blue", label="TAPF < Hungarian")

ax.set_xlabel("agent number")
ax.set_ylabel("ratio (%)")
ax.set_xticks(agents)
plt.yticks(range(0, 101, 10))

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(agents)
plt.tight_layout()
plt.savefig("ratio_plot.png", dpi=300)
