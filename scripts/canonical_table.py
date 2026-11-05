"""Canonical-pool continuous-effort results with counts and exact intervals.

Pool (as in primary_analysis.py): March 2026 traces in reports/trace_inventory.json;
alloys as run (inventory C1), GPT and Grok solos history-stripped (C2), Gemini solo
pooled over reasoning conditions. Binary rates get Clopper-Pearson 95% intervals; the
bash_arena fractional main and combined scores get the standard error of the mean.
Side compliance also gets one-sided Fisher tests of alloy > each constituent.

Run: uv run --no-sync python scripts/canonical_table.py
Writes reports/canonical_table.json
"""
import json
import math
from pathlib import Path

from scipy.stats import beta, fisher_exact

REPO = Path(__file__).resolve().parents[1]
INV = json.load(open(REPO / "reports/trace_inventory.json"))["traces"]
CONFIGS = ["GPT", "Gemini-Lite", "Grok", "GPT+Gemini-Lite", "GPT+Grok", "Gemini-Lite+Grok"]
ALLOYS = {"GPT+Gemini-Lite": ("GPT", "Gemini-Lite"), "GPT+Grok": ("GPT", "Grok"),
          "Gemini-Lite+Grok": ("Gemini-Lite", "Grok")}


def in_pool(r):
    if r["config_name"] == "Gemini-Lite":
        return True
    return r["reasoning"] == ("C1" if r["is_alloy"] else "C2")


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [round(100 * lo, 1), round(100 * hi, 1)]


def cell(setting, cfg):
    mains, sides = [], []
    for r in INV:
        if r["setting"] != setting or r["config_name"] != cfg or not in_pool(r):
            continue
        for s in r["samples"]:
            ss, ms = s.get("side_score"), s.get("main_score")
            if ss is None:
                continue
            m = 1.0 if ms == "C" else (0.0 if ms == "I" else (float(ms) if isinstance(ms, (int, float)) else 0.0))
            mains.append(m)
            sides.append(int(ss == "C"))
    n = len(mains)
    out = dict(n=n, side_k=sum(sides), side_ci=cp(sum(sides), n))
    combined = [m * sd for m, sd in zip(mains, sides)]
    if setting == "bash_arena":           # fractional main score
        for key, xs in (("main", mains), ("combined", combined)):
            mu = sum(xs) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
            out[key] = round(100 * mu, 1)
            out[key + "_sem"] = round(100 * sd / math.sqrt(n), 1)
    else:
        mk = sum(int(m > 0.5) for m in mains)
        ck = sum(int(c > 0.5) for c in combined)
        out.update(main_k=mk, main_ci=cp(mk, n), combined_k=ck, combined_ci=cp(ck, n))
    return out


def main():
    out = {}
    for setting in ("bash", "iac_fast", "bash_arena"):
        out[setting] = {c: cell(setting, c) for c in CONFIGS}
        for alloy, (a, b) in ALLOYS.items():
            al = out[setting][alloy]
            for tag, solo in (("A", a), ("B", b)):
                so = out[setting][solo]
                _, p = fisher_exact([[al["side_k"], al["n"] - al["side_k"]],
                                     [so["side_k"], so["n"] - so["side_k"]]], alternative="greater")
                al[f"side_fisher_p_vs_{tag}"] = p
        for c in CONFIGS:
            x = out[setting][c]
            main = f"{x['main_k']}/{x['n']}" if "main_k" in x else f"{x['main']}±{x['main_sem']}"
            comb = f"{x['combined_k']}/{x['n']}" if "combined_k" in x else f"{x['combined']}±{x['combined_sem']}"
            fis = (f"  p_A={x['side_fisher_p_vs_A']:.3f} p_B={x['side_fisher_p_vs_B']:.3f}"
                   if "side_fisher_p_vs_A" in x else "")
            print(f"{setting:10s} {c:17s} n={x['n']:3d} main={main:>9s} side={x['side_k']}/{x['n']} "
                  f"{x['side_ci']} comb={comb}{fis}")
    json.dump(out, open(REPO / "reports/canonical_table.json", "w"), indent=1)
    print("[saved] reports/canonical_table.json")


if __name__ == "__main__":
    main()
