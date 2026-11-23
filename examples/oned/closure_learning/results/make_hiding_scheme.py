"""Write the SVG diagram of the idea of a hiding function: (a) grains of different size on
one bed, a small one sheltered and a large one exposed; (b) the reference shear stress of
a size relative to that of the mean size, between the limits of no hiding and of equal
mobility, with the function of Wilcock and Crowe (2003).

The equation is typeset by matplotlib (as outlines); everything else is plain SVG shapes
and text. Argument: output file.
"""
import io
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 1240, 520
BLACK, ACCENT = "#000000", "#a61c1c"
FINE_FILL, COARSE_FILL, GRAIN_FILL = "#f4cfc6", "#dfe9f3", "#f1efe9"
FONT = "Arial, Helvetica, sans-serif"
plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path", "svg.hashsalt": "hiding-scheme"})

fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=72)
item = fig.text(972 / W, 1.0 - 332 / H, r"$\tau_{r,i} = \tau_{r,m}\left(\dfrac{D_i}{D_m}\right)^{b}$", fontsize=20, ha="left", va="baseline")
buffer = io.StringIO()
fig.savefig(buffer, format="svg", transparent=True)
svg = buffer.getvalue()

under, over = [], []


def text(x, y, content, size=14, weight="normal", anchor="start", extra=""):
    over.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{BLACK}" '
                f'text-anchor="{anchor}"{extra}>{content}</text>')


def line(points, width=1.2, dash="", color=BLACK, target=None):
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    (over if target is None else target).append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dashed}/>')


def arrow(points, width=1.3, color=BLACK):
    (x1, y1), (x2, y2) = points[-2], points[-1]
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    head, half = 9.0, 3.8
    bx, by = x2 - head * ux, y2 - head * uy
    line(list(points[:-1]) + [(bx + ux, by + uy)], width, "", color)
    over.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx - half * uy:.1f},{by + half * ux:.1f} {bx + half * uy:.1f},{by - half * ux:.1f}" fill="{color}"/>')


def grain(x, y, r, fill=GRAIN_FILL, stroke=BLACK, width=1.1):
    over.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


under.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

# ------------------------------------------------------------------ (a) the bed
text(24, 34, "(a) Mixed sizes on one bed", 19, "bold")
text(62, 96, "flow", 14)
line([(62, 106), (62, 218)], 1.0)
tips = []
for y, length in ((116, 118), (140, 104), (164, 86), (188, 62), (210, 30)):
    arrow([(62, y), (62 + length, y)])
    tips.append((62 + length, y))
# Deposit below the surface, then the surface grains on top.
for x, y, r in ((80, 461, 30), (150, 464, 26), (215, 458, 24), (280, 466, 30), (345, 460, 26), (410, 464, 30), (478, 460, 28),
                (542, 464, 26), (116, 438, 7), (183, 440, 6), (248, 437, 6), (313, 440, 7), (378, 438, 6), (445, 437, 6), (512, 440, 6)):
    grain(x, y, r)
for x, y, r in ((130, 372, 58), (292, 380, 50), (56, 406, 20), (364, 400, 22), (540, 402, 24), (236, 420, 8), (194, 422, 7),
                (338, 424, 8), (390, 426, 7), (26, 432, 8), (572, 430, 8), (512, 426, 8)):
    grain(x, y, r)
grain(452, 364, 66, COARSE_FILL, BLACK, 1.6)             # the exposed large grain
grain(214, 404, 13, FINE_FILL, ACCENT, 1.8)             # the sheltered small grain
for k, words in enumerate(("small grain: sheltered by its", "neighbours, harder to move", "than on a bed of its own size")):
    text(214, 250 + 18 * k, words, 14, anchor="middle")
arrow([(214, 296), (214, 388)], 1.2)
for k, words in enumerate(("large grain: exposed to the flow,", "easier to move than on a bed", "of its own size")):
    text(452, 216 + 18 * k, words, 14, anchor="middle")
arrow([(452, 262), (452, 295)], 1.2)

# ------------------------------------------------------------------ (b) the function
line([(600, 16), (600, H - 16)], 1.0, "3 5")
text(624, 34, "(b) Hiding function", 19, "bold")
left, right, top, bottom = 724, 1184, 70, 430


def px(x):
    return left + (right - left) / 2.0 * (math.log10(x) + 1.0)


def py(y):
    return bottom - (bottom - top) / 2.0 * (math.log10(y) + 1.0)


def published(x):
    """tau_r,i / tau_r,m of Wilcock and Crowe (2003)."""
    return x ** (0.67 / (1.0 + math.exp(1.5 - x)))


samples = [10 ** (-1.0 + 2.0 * k / 120) for k in range(121)]
curve = [(px(x), py(published(x))) for x in samples]
fine = [p for p, x in zip(curve, samples) if x <= 1.0]
coarse = [p for p, x in zip(curve, samples) if x >= 1.0]
# Shaded: between the line of no hiding and the published function.
under.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in fine + [(px(1), py(1)), (px(0.1), py(0.1))]) + f'" fill="{FINE_FILL}"/>')
under.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in coarse + [(px(10), py(10)), (px(1), py(1))]) + f'" fill="{COARSE_FILL}"/>')
line([(left, top), (left, bottom), (right, bottom)], 1.3)
for value, label in ((0.1, "0.1"), (1.0, "1"), (10.0, "10")):
    line([(px(value), bottom), (px(value), bottom + 6)], 1.2)
    line([(left - 6, py(value)), (left, py(value))], 1.2)
    text(px(value), bottom + 22, label, 13.5, anchor="middle")
    text(left - 10, py(value) + 5, label, 13.5, anchor="end")
for value in (0.2, 0.5, 2.0, 5.0):
    line([(px(value), bottom), (px(value), bottom + 3.5)], 1.0)
    line([(left - 3.5, py(value)), (left, py(value))], 1.0)
text((left + right) / 2, bottom + 46, "grain size / mean size of the bed surface", 14.5, anchor="middle")
text(674, (top + bottom) / 2, "reference shear stress / that of the mean size", 14.5, anchor="middle",
     extra=f' transform="rotate(-90 674 {(top + bottom) / 2})"')
line([(px(0.1), py(0.1)), (px(10), py(10))], 1.3, "7 4")
line([(px(0.1), py(1)), (px(10), py(1))], 1.3, "2 4")
line(curve, 2.8)
angle = -math.degrees(math.atan2(bottom - top, right - left))
text(px(4.6) - 16, py(4.6) - 14, "no hiding", 14, anchor="middle", extra=f' transform="rotate({angle:.1f} {px(4.6) - 16:.1f} {py(4.6) - 14:.1f})"')
text(px(0.105), py(1) - 8, "equal mobility", 14)
slope = -math.degrees(math.atan(0.66 * (bottom - top) / (right - left)))
text(1090, 232, "Wilcock-Crowe", 14, "bold", extra=f' transform="rotate({slope:.1f} 1090 232)"')
# What the two shaded parts mean.
for k, (fill, first, second) in enumerate(((FINE_FILL, "finer than the mean: hidden,", "harder to move than alone"),
                                           (COARSE_FILL, "coarser than the mean: exposed,", "easier to move than alone"))):
    y0 = 98 + 52 * k
    over.append(f'<rect x="742" y="{y0 - 12}" width="16" height="16" fill="{fill}" stroke="{BLACK}" stroke-width="0.9"/>')
    text(766, y0, first, 13.5)
    text(766, y0 + 17, second, 13.5)
# The exponent.
text(972, 372, "b = 1: no hiding", 13.5)
text(972, 391, "b = 0: equal mobility", 13.5)
text(972, 410, "Wilcock-Crowe: b = 0.12 to 0.67", 13.5)

marker = '<g id="figure_1">'
assert svg.count(marker) == 1
svg = svg.replace(marker, "\n".join(under) + "\n" + marker)
svg = svg.replace("</svg>", "\n".join(over) + "\n</svg>")
open(sys.argv[1], "w").write(svg)
print("wrote", sys.argv[1])
