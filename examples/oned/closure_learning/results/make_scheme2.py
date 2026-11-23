"""Write the SVG diagram (plain block-diagram style, network drawn as a network) of how
one hiding network is trained on three data sets and tested on two others."""
import sys

W, H = 1240, 566
DOWN = 26                           # room for the panel headings above the diagram
BLACK = "#000000"
FONT = "Arial, Helvetica, sans-serif"
SOLVER_FILL = "#dfe9f3"
NODE_FILL = "#f6c9ab"
LOOP = "#a61c1c"
out = []


def rect(x, y, w, h, fill="#ffffff", width=1.2):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{BLACK}" stroke-width="{width}"/>')


def text(x, y, content, size=14, weight="normal", anchor="middle", extra=""):
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{BLACK}" '
               f'text-anchor="{anchor}"{extra}>{content}</text>')


def italic(x, y, content, size=15):
    text(x, y, content, size, anchor="start", extra=' font-style="italic"')


def box(x, y, w, h, lines, fill="#ffffff", title_size=15, line_size=13.5, width=1.2):
    """A box with a bold first line and plain further lines, centred."""
    rect(x, y, w, h, fill, width)
    step = line_size + 5.5
    total = title_size + (len(lines) - 1) * step
    top = y + (h - total) / 2 + title_size - 3
    text(x + w / 2, top, lines[0], title_size, "bold")
    for k, line in enumerate(lines[1:], start=1):
        text(x + w / 2, top + k * step, line, line_size)


def line(points, color=BLACK, width=1.4, dash=""):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dashed}/>')


def arrow(points, color=BLACK, width=1.4, dash=""):
    """A line through the points with a drawn arrowhead at its end."""
    (x1, y1), (x2, y2) = points[-2], points[-1]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    head, half = 10.0, 4.2
    bx, by = x2 - head * ux, y2 - head * uy
    line(list(points[:-1]) + [(round(bx + ux, 1), round(by + uy, 1))], color, width, dash)
    tip = [(x2, y2), (bx - half * uy, by + half * ux), (bx + half * uy, by - half * ux)]
    out.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in tip) + f'" fill="{color}"/>')


def network(cx, cy, spacing, n_hidden=8, pitch=13.0, radius=4.6):
    """A 1-8-8-1 network: connections first, then the nodes on top."""
    xs = [cx - 1.5 * spacing, cx - 0.5 * spacing, cx + 0.5 * spacing, cx + 1.5 * spacing]
    hidden = [cy + (k - (n_hidden - 1) / 2) * pitch for k in range(n_hidden)]
    layers = [[cy], hidden, hidden, [cy]]
    for a in range(3):
        for y1 in layers[a]:
            for y2 in layers[a + 1]:
                out.append(f'<line x1="{xs[a]}" y1="{y1}" x2="{xs[a + 1]}" y2="{y2}" stroke="#4d4d4d" stroke-width="0.55"/>')
    for a, ys in enumerate(layers):
        for y in ys:
            end = a in (0, 3)
            out.append(f'<circle cx="{xs[a]}" cy="{y}" r="{radius + 1.4 if end else radius}" fill="{"#ffffff" if end else NODE_FILL}" '
                       f'stroke="{BLACK}" stroke-width="1.1"/>')
    # Input d_i / d_g and output xi_i: letters and subscripts placed one by one.
    x0, y0 = xs[0] - 13 - 42, cy + 5
    italic(x0, y0, "d")
    italic(x0 + 8.8, y0 + 4.5, "i", 10.5)
    italic(x0 + 14.5, y0, "/")
    italic(x0 + 23.0, y0, "d")
    italic(x0 + 31.8, y0 + 4.5, "g", 10.5)
    italic(xs[3] + 13, y0, "&#958;", 16)
    italic(xs[3] + 21.5, y0 + 4.5, "i", 10.5)


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

# ------------------------------------------------------------------ training
text(24, 34, "(a) Training", 19, "bold", "start")
net = (225, 56 + DOWN, 350, 122)                            # x, y, width, height of the network frame
text(net[0] + net[2] / 2, net[1] - 10, "Hiding function as a neural network", 15, "bold")
rect(*net)
network(net[0] + net[2] / 2, net[1] + net[3] / 2, 62)

centres = (140, 400, 660)
box_w, model_y, model_h = 224, 226 + DOWN, 82
models = (
    (["Flume run 2 (33% sand)", "differentiable 1-D solver", "Seal et al. (1997)"], SOLVER_FILL),
    (["Flume run 6 (53% sand)", "differentiable 1-D solver", "Toro-Escobar et al. (2000)"], SOLVER_FILL),
    (["Transport measurements", "formula only, no simulation", "Wilcock et al. (2001)"], "#ffffff"),
)
misfit_y, misfit_h = 338 + DOWN, 54
misfits = (
    ["Misfit 1", "thickness, grain sizes, depth"],
    ["Misfit 2", "thickness, grain sizes, depth"],
    ["Misfit 3", "transport rate of each size"],
)
bus = 202 + DOWN
line([(net[0] + net[2] / 2, net[1] + net[3]), (net[0] + net[2] / 2, bus)])
line([(centres[0], bus), (centres[2], bus)])
for k, cx in enumerate(centres):
    lines, fill = models[k]
    arrow([(cx, bus), (cx, model_y)])
    box(cx - box_w / 2, model_y, box_w, model_h, lines, fill)
    arrow([(cx, model_y + model_h), (cx, misfit_y)])
    box(cx - box_w / 2, misfit_y, box_w, misfit_h, misfits[k])
bus2 = 414 + DOWN
for cx in centres:
    line([(cx, misfit_y + misfit_h), (cx, bus2)])
line([(centres[0], bus2), (centres[2], bus2)])
opt = (230, 440 + DOWN, 340, 62)
arrow([(centres[1], bus2), (centres[1], opt[1])])
box(*opt, ["Minimize misfit 1 + 2 + 3", "gradients by automatic differentiation"], "#ffffff", 15, 13.5, 1.8)
# Feedback loop.
loop_x = 14
arrow([(opt[0], opt[1] + opt[3] / 2), (loop_x, opt[1] + opt[3] / 2), (loop_x, net[1] + net[3] / 2), (net[0], net[1] + net[3] / 2)],
      LOOP, 1.6, "7 4")
text(122, opt[1] + opt[3] / 2 - 9, "update weights", 14)

# ------------------------------------------------------------------ testing
divider = 826
line([(divider, 16), (divider, H - 16)], BLACK, 1.0, "3 5")
text(divider + 24, 34, "(b) Testing", 19, "bold", "start")
net2 = (900, 56 + DOWN, 280, 122)
text(net2[0] + net2[2] / 2, net2[1] - 10, "Trained network (weights fixed)", 15, "bold")
rect(*net2)
network(net2[0] + net2[2] / 2, net2[1] + net2[3] / 2, 46)
arrow([(net[0] + net[2], net[1] + net[3] / 2), (net2[0], net2[1] + net2[3] / 2)])
text((net[0] + net[2] + divider) / 2, net[1] + net[3] / 2 - 9, "after training", 14)

test_centres = (948, 1132)
test_w = 172
tests = (
    (["Flume runs 1 and 3", "1-D solver", "Seal et al. (1997)"], SOLVER_FILL),
    (["Other sediments", "formula only", "Wilcock et al. (2001)"], "#ffffff"),
)
results = (
    ["Thickness error", "4.4 and 5.0 cm", "published: 5.0, 5.9 cm"],
    ["Transport rates", "scatter 0.52 (log10)", "published: 0.49"],
)
line([(net2[0] + net2[2] / 2, net2[1] + net2[3]), (net2[0] + net2[2] / 2, bus)])
line([(test_centres[0], bus), (test_centres[1], bus)])
for k, cx in enumerate(test_centres):
    lines, fill = tests[k]
    arrow([(cx, bus), (cx, model_y)])
    box(cx - test_w / 2, model_y, test_w, model_h, lines, fill)
    arrow([(cx, model_y + model_h), (cx, misfit_y)])
    box(cx - test_w / 2, misfit_y, test_w, 76, results[k], "#ffffff", 15, 13)

out.append("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[1])
