# https://pythonforthelab.com/blog/python-tip-ready-publish-matplotlib-figures/
import matplotlib
from matplotlib import ticker

matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

import matplotlib.pyplot as plt
import numpy as np

xf = np.linspace(0, 10, 100)
yf = 5.0 * np.sin(xf)
zf = 0.0005 * np.exp(xf)

af = 0.5 + yf * np.cos(xf)
bf = af + zf
cf = bf + af + yf


# get desired width and height from pdf in inches
wd = 3.19423 * 2  # inches
ht = 3.19423  # inches
fig = plt.figure(figsize=(wd, ht), frameon=True, layout="tight", dpi=600)
ax = plt.subplot(1, 1, 1)
# ax.set_aspect("equal")
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.axhline(color="gray", alpha=0.4)
# ax.axvline(color="gray", alpha=0.4)
ax.grid(True)
ax.grid(axis="y")  # or x and y specificly

# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xticks([])
# ax.set_yticks([])
ax.plot(xf, yf, label="label curve")
ax.plot(xf, zf, label="expo")
ax.plot(xf, af, label="af")
ax.plot(xf, bf, label="bf")
ax.plot(xf, cf, label="cf")
ax.set_xlabel("xlabel", fontsize=12)
ax.set_ylabel("ylabel", fontsize=12)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", length=5, width=1, colors="black", grid_color="gray", grid_alpha=0.2, labelsize=12)
ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1, colors="black", grid_color="gray", grid_alpha=0.1, labelsize=12)
ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", ncols=3, mode="expand", borderaxespad=0.0, edgecolor="k", fancybox=False, fontsize=12)
ax.set_xmargin(0.01)
ax.set_ymargin(0.01)

plt.savefig("/home/yuth/paper_format.svg", bbox_inches="tight")



x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, label="sin wave")
ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", ncols=2, mode="expand", borderaxespad=0.0)
plt.show()

# https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib
# this mean place the lower left of the legend box to location of x=0, y=1.02, with width=1.0, height=0.102