"""Write the SVG diagram of the differentiable 1-D solver with a neural-network hiding
function: the network on top, the equations of flow, sediment and bed and the closures
below, and an arrow from the output of the network to the place where it enters.

The equations are typeset by matplotlib (as outlines, so that they look the same in every
program); boxes, the network and the labels are plain SVG shapes and text.
Argument: output file.
"""
import io
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 1240, 790
BLACK, ACCENT = "#000000", "#a61c1c"
FLOW_FILL, NODE_FILL = "#dfe9f3", "#f6c9ab"
FONT = "Arial, Helvetica, sans-serif"
plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path", "svg.hashsalt": "closure-scheme"})

fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=72)
renderer = fig.canvas.get_renderer()


def equation(x, y, tex, size=21, color=BLACK):
    """Place an equation with its left end at x and its baseline at y (pixels from the
    top left); returns its box (x0, y0, x1, y1) in the same coordinates."""
    item = fig.text(x / W, 1.0 - y / H, f"${tex}$", fontsize=size, color=color, ha="left", va="baseline")
    box = item.get_window_extent(renderer)
    return box.x0, H - box.y1, box.x1, H - box.y0


# ---------------------------------------------------------------- equations
left, right = 58, 672
boxes = {}
boxes["mass"] = equation(left, 342, r"\dfrac{\partial A}{\partial t} + \dfrac{\partial Q}{\partial x} = 0")
boxes["friction"] = equation(400, 342, r"S_f = \dfrac{Q\,|Q|}{K^2}")
boxes["momentum"] = equation(
    left, 412, r"\dfrac{\partial Q}{\partial t} + \dfrac{\partial}{\partial x}\left(\dfrac{\beta Q^2}{A}\right)"
    r" + gA\,\dfrac{\partial \eta}{\partial x} + gA\,S_f = 0")
boxes["load"] = equation(left, 534, r"\dfrac{\partial Q_i}{\partial x} = \dfrac{Q_i^* - Q_i}{L_i}")
boxes["exner"] = equation(left, 616, r"(1-\lambda)\,B\,\dfrac{\partial z_b}{\partial t} = -\sum_i \dfrac{\partial Q_i}{\partial x}")
boxes["layer"] = equation(
    left, 712, r"(1-\lambda)\,B\left[f_{I,i}\,\dfrac{\partial (z_b - L_a)}{\partial t} + \dfrac{\partial (L_a F_i)}{\partial t}\right]"
    r" = -\dfrac{\partial Q_i}{\partial x}", 19)
boxes["reference"] = equation(right, 352, r"\tau_{r,i} = \tau_{r,m}\;\dfrac{d_i}{d_g}")
boxes["xi"] = equation(boxes["reference"][2] + 10, 352, r"\xi_i^{-1}", 26, ACCENT)
boxes["published"] = equation(right, 428, r"\xi_i = (d_i / d_g)^{\,1-b}, \quad b = \dfrac{0.67}{1+\exp(1.5 - d_i/d_g)}", 15)
boxes["curve"] = equation(right, 516, r"\phi_i = \dfrac{\tau_b}{\tau_{r,i}}, \qquad W_i^* = W^*(\phi_i)")
boxes["capacity"] = equation(right, 606, r"Q_i^* = B\,F_i\,W_i^*\,\dfrac{u_*^3}{(s-1)\,g}")
boxes["transfer"] = equation(right, 706, r"f_{I,i} = c\,p_i + (1-c)\,F_i")

buffer = io.StringIO()
fig.savefig(buffer, format="svg", transparent=True)
svg = buffer.getvalue()
for name, b in boxes.items():
    print(f"{name:10s} x {b[0]:7.1f} to {b[2]:7.1f}, y {b[1]:6.1f} to {b[3]:6.1f}")

# ---------------------------------------------------------------- shapes and labels
under, over = [], []


def rect(target, x, y, w, h, fill="none", width=1.2):
    target.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{BLACK}" stroke-width="{width}"/>')


def text(x, y, content, size=14, weight="normal", anchor="start", extra="", fill=BLACK):
    over.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
                f'text-anchor="{anchor}"{extra}>{content}</text>')


def italic(x, y, content, size=15, fill=BLACK):
    text(x, y, content, size, extra=' font-style="italic"', fill=fill)


def note(name, lines, gap=24):
    """Plain words to the right of an equation, centred on its height."""
    b = boxes[name]
    middle = (b[1] + b[3]) / 2
    first = middle + 5 - 9 * (len(lines) - 1)
    for k, line in enumerate(lines):
        text(b[2] + gap, first + 18 * k, line, 14)


under.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
text(W / 2, 34, "Differentiable 1-D universal shallow-water solver with sediment transport", 21, "bold", "middle")
rect(over, 14, 50, W - 28, H - 64, "none", 1.4)

# The network.
net = (380, 84, 480, 168)
text(net[0] + net[2] / 2, net[1] - 9, "Hiding function as a neural network", 15, "bold", "middle")
rect(over, *net)
cx, cy, spacing, pitch, radius = net[0] + net[2] / 2, net[1] + net[3] / 2 + 13, 70, 13.0, 4.6
xs = [cx - 1.5 * spacing, cx - 0.5 * spacing, cx + 0.5 * spacing, cx + 1.5 * spacing]
hidden = [cy + (k - 3.5) * pitch for k in range(8)]
layers = [[cy], hidden, hidden, [cy]]
for a in range(3):
    for y1 in layers[a]:
        for y2 in layers[a + 1]:
            over.append(f'<line x1="{xs[a]}" y1="{y1}" x2="{xs[a + 1]}" y2="{y2}" stroke="#4d4d4d" stroke-width="0.55"/>')
for a, ys in enumerate(layers):
    for y in ys:
        end = a in (0, 3)
        over.append(f'<circle cx="{xs[a]}" cy="{y}" r="{radius + 1.6 if end else radius}" fill="{"#ffffff" if end else NODE_FILL}" '
                    f'stroke="{ACCENT if a == 3 else BLACK}" stroke-width="{1.6 if a == 3 else 1.1}"/>')
label_y = net[1] + 24
text(xs[0], label_y, "Input", 15, anchor="middle")
text(cx, label_y, "Hidden layers", 15, anchor="middle")
text(xs[3], label_y, "Output", 15, anchor="middle")
x0, y0 = xs[0] - 13 - 46, cy + 6
italic(x0, y0, "d", 17)
italic(x0 + 10.0, y0 + 5, "i", 12)
italic(x0 + 16.5, y0, "/", 17)
italic(x0 + 26.0, y0, "d", 17)
italic(x0 + 36.0, y0 + 5, "g", 12)
italic(xs[3] + 14, y0, "&#958;", 19, ACCENT)
italic(xs[3] + 24, y0 + 5, "i", 12, ACCENT)

# Frames and headings of the equation groups.
rect(under, 34, 274, 596, 168, FLOW_FILL)
rect(under, 34, 458, 596, 304, "#ffffff")
rect(under, 648, 274, 564, 488, "#ffffff")
text(48, 298, "Flow: shallow-water equations", 15, "bold")
text(48, 482, "Sediment transport and bed, size class i", 15, "bold")
text(662, 298, "Closures", 15, "bold")
note("load", ["load"])
note("exner", ["bed elevation"])
note("layer", ["surface", "layer"], 16)
note("xi", ["reference shear stress, with the", "hiding factor from the network"])
note("curve", ["transport function"])
note("capacity", ["transport capacity", "(Wilcock and Crowe, 2003)"])
note("transfer", ["exchange between surface", "layer and deposit"])
b = boxes["published"]
text(b[0], b[1] - 8, "in place of the published hiding function", 14)

# Arrow from the output of the network to the hiding factor in the reference stress.
start = (xs[3] + 44, cy + 2)
end = ((boxes["xi"][0] + boxes["xi"][2]) / 2 - 4, boxes["xi"][1] - 5)
c1, c2 = (start[0] + 95, start[1] - 2), (end[0] + 48, end[1] - 120)
dx, dy = end[0] - c2[0], end[1] - c2[1]
norm = (dx * dx + dy * dy) ** 0.5
ux, uy = dx / norm, dy / norm
head, half = 12.0, 5.0
bx, by = end[0] - head * ux, end[1] - head * uy
over.append(f'<path d="M {start[0]} {start[1]} C {c1[0]} {c1[1]}, {c2[0]:.1f} {c2[1]:.1f}, {bx + ux:.1f} {by + uy:.1f}" fill="none" '
            f'stroke="{ACCENT}" stroke-width="1.8"/>')
over.append(f'<polygon points="{end[0]:.1f},{end[1]:.1f} {bx - half * uy:.1f},{by + half * ux:.1f} {bx + half * uy:.1f},{by - half * ux:.1f}" fill="{ACCENT}"/>')

# ---------------------------------------------------------------- assemble
marker = '<g id="figure_1">'
assert svg.count(marker) == 1
svg = svg.replace(marker, "\n".join(under) + "\n" + marker)
svg = svg.replace("</svg>", "\n".join(over) + "\n</svg>")
open(sys.argv[1], "w").write(svg)
print("wrote", sys.argv[1])
