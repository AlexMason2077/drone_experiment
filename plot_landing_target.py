import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CSV_FILE = "tello_landing_points.csv"

df = pd.read_csv(CSV_FILE)

x = df["x_after"]
y = df["y_after"]

fig, ax = plt.subplots(figsize=(8, 8))

# scatter landing points
ax.scatter(x, y, s=50, alpha=0.8, label="Landing Points")

# mark center of mission pad
ax.scatter(0, 0, s=120, marker="x", label="Mission Pad Center")

# add concentric circles like a target
radii = [5, 10, 15, 20]
for r in radii:
    circle = Circle((0, 0), r, fill=False)
    ax.add_patch(circle)

# annotate trial number
for i, row in df.iterrows():
    ax.text(row["x_after"] + 0.3, row["y_after"] + 0.3,str(i+1), fontsize=9)

ax.set_xlabel("X offset from pad center (cm)")
ax.set_ylabel("Y offset from pad center (cm)")
ax.set_title("Tello Landing Points Relative to Mission Pad Center")

ax.axhline(0)
ax.axvline(0)
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

# auto range
max_abs = max(
    max(abs(x).max(), abs(y).max(), 10),
    30
)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)

plt.show()