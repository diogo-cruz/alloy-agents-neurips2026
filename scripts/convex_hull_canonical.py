"""Appendix tab:convex_hull on the canonical pool (reports/canonical_table.json).

For each alloy, in main-task x side-task space, on point estimates:
  - perpendicular distance to the line through its two solos (unsigned);
  - whether it is closer to (100,100) than the midpoint of those solos;
  - whether some mixture of its own two solos, or of all three solos of the setting, is at least as
    high on both rates. A point no mixture matches lies beyond that solo frontier.
Descriptive only: no intervals, and the solo points are themselves uncertain.

Run: uv run python scripts/convex_hull_canonical.py
Writes reports/convex_hull_canonical.json and prints the LaTeX rows.
"""
import json
import math
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLOS = ["GPT", "Gemini-Lite", "Grok"]
ALLOYS = ["GPT+Gemini-Lite", "GPT+Grok", "Gemini-Lite+Grok"]
TEX = {"GPT+Gemini-Lite": "GPT+Gemini", "GPT+Grok": "GPT+Grok", "Gemini-Lite+Grok": "Gemini+Grok"}
SETTING_TEX = {"bash": "Bash", "iac_fast": "iac\\_fast", "bash_arena": "bash\\_arena"}
EPS = 1e-9


def point(row):
    """(main %, side %). bash_arena's main rate is already a mean fractional score in percent."""
    main = row["main"] if "main" in row else 100 * row["main_k"] / row["n"]
    return main, 100 * row["side_k"] / row["n"]


def perp_distance(p, a, b):
    (x, y), (x1, y1), (x2, y2) = p, a, b
    return abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / math.hypot(x2 - x1, y2 - y1)


def to_corner(p):
    return math.hypot(100 - p[0], 100 - p[1])


def dominated(p, vertices):
    """True if a mixture of the vertices is >= p on both rates.

    The set of mixtures is a convex polygon, so it is enough to test the vertices and, for each
    pair of vertices, the two points of that segment that meet p's main rate and p's side rate.
    """
    cands = list(vertices)
    for a, b in combinations(vertices, 2):
        for axis in (0, 1):
            if abs(b[axis] - a[axis]) > EPS:
                t = (p[axis] - a[axis]) / (b[axis] - a[axis])
                if 0 <= t <= 1:
                    cands.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return any(c[0] >= p[0] - EPS and c[1] >= p[1] - EPS for c in cands)


def main():
    table = json.load(open(REPO / "reports/canonical_table.json"))
    rows = []
    for setting in ("bash", "bash_arena", "iac_fast"):
        solos = {s: point(table[setting][s]) for s in SOLOS}
        for alloy in ALLOYS:
            a, b = alloy.split("+")
            p = point(table[setting][alloy])
            mid = ((solos[a][0] + solos[b][0]) / 2, (solos[a][1] + solos[b][1]) / 2)
            rows.append(dict(
                setting=setting, alloy=alloy,
                main=round(p[0], 1), side=round(p[1], 1),
                perp_distance=round(perp_distance(p, solos[a], solos[b]), 1),
                closer_than_midpoint=to_corner(p) < to_corner(mid),
                beyond_own_two_solos=not dominated(p, [solos[a], solos[b]]),
                beyond_all_three_solos=not dominated(p, list(solos.values())),
            ))
    summary = {k: sum(r[k] for r in rows) for k in
               ("closer_than_midpoint", "beyond_own_two_solos", "beyond_all_three_solos")}
    out = dict(source="reports/canonical_table.json", rows=rows, of_nine=summary)
    json.dump(out, open(REPO / "reports/convex_hull_canonical.json", "w"), indent=1)

    yn = lambda v: "Yes" if v else "No"
    last = None
    for r in rows:
        if last and r["setting"] != last:
            print("\\midrule")
        last = r["setting"]
        print(f"{SETTING_TEX[r['setting']]} & {TEX[r['alloy']]} & {r['perp_distance']:.1f} & "
              f"{yn(r['closer_than_midpoint'])} & {yn(r['beyond_own_two_solos'])} & "
              f"{yn(r['beyond_all_three_solos'])} \\\\")
    print(summary)


if __name__ == "__main__":
    main()
