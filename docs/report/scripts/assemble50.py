"""Second-hull tables and statistics, appended to the report build."""
import json, pathlib
B = pathlib.Path("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad")
from wave_resistance.reference import hull_reference

hc = json.load(open(B / "hydro_compare.json"))
ratio = json.load(open(B / "ratio.json"))
load = lambda n: json.load(open(B / n))
_d01 = load("s280_static.json")
s01 = {round(r["fn"], 2): r for r in _d01["records"] if r.get("panels")}
# The Sysser 01 static sweep predates the per-point hydrostatics, and at the static
# attitude the displaced volume is the reference volume by construction.
for _r in s01.values():
    _r.setdefault("volume_at_attitude", _d01["volume"])
d50 = load("s50_static.json")
s50 = {round(r["fn"], 2): r for r in d50["records"] if r.get("panels")}
piv50 = {round(r["fn"], 2): r for r in load("s50_pivot.json")["records"] if r.get("panels")} \
    if (B / "s50_pivot.json").exists() else {}
piv01 = {round(r["fn"], 2): r for r in load("s01_pivot.json")["records"] if r.get("panels")} \
    if (B / "s01_pivot.json").exists() else {}

r1, r50 = hull_reference(1), hull_reference(50)
out = {}

# ---- table of hull particulars ----
rows = []
for lab, key, unit, fmt in (("waterline length", "lwl", r"\si{\metre}", "{:.4f}"),
                            ("waterline beam", "bwl", r"\si{\metre}", "{:.4f}"),
                            ("canoe-body draught", "tc", r"\si{\metre}", "{:.4f}"),
                            ("displaced volume", "volume", r"\si{\metre\cubed}", "{:.6f}"),
                            ("wetted area", "wetted_area", r"\si{\metre\squared}", "{:.4f}"),
                            ("prismatic coefficient", "cp", "---", "{:.4f}"),
                            ("midship coefficient", "cm", "---", "{:.4f}")):
    rows.append(f"{lab} & {unit} & {fmt.format(r1[key])} & {fmt.format(r50[key])} \\\\")
rows.append(r"beam / draught & --- & "
            f"{r1['bwl'] / r1['tc']:.2f} & {r50['bwl'] / r50['tc']:.2f} \\\\")
rows.append(r"LCB aft of midships & \si{\percent} & "
            f"{100 * r1['lcb'] / r1['lwl']:.2f} & {100 * r50['lcb'] / r50['lwl']:.2f} \\\\")
rows.append(r"displacement weight & \si{\newton} & "
            f"{1000 * 9.80665 * r1['volume']:.1f} & {1000 * 9.80665 * r50['volume']:.1f} \\\\")
out["TABHULLS"] = (r"""\begin{table}[htbp]
\centering
\caption{Published model-scale particulars of the two hulls, from the DSYHS hydrostatics
release. Sysser 01 is the parent of series 1 and Sysser 50 belongs to series 4.}
\label{tab:hulls}
\small
\begin{tabular}{@{}llS[table-format=1.6]S[table-format=1.6]@{}}
\toprule
& unit & {Sysser 01} & {Sysser 50} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# ---- geometry discrepancy table ----
g = []
for key, lab in (("lwl", r"$c$"), ("bwl", r"$b$"), ("tc", r"$d$"),
                 ("wetted_area", r"$S_c$"), ("waterplane_area", r"$A_w$"),
                 ("midship_area", r"$A_m$"), ("volume", r"$\nabla_c$")):
    a = hc["sysser01"]["rows"][key]
    b = hc["sysser50"]["rows"][key]
    g.append(f"{lab} & {a['published']:.6f} & \\num{{{a['pct']:+.3f}}} & "
             f"{b['published']:.6f} & \\num{{{b['pct']:+.3f}}} \\\\")
g.append(r"$\mathrm{LCB}$ & "
         f"{hc['sysser01']['lcb']['published']:.6f} & "
         f"\\num{{{hc['sysser01']['lcb']['mm']:+.2f}}}\\,mm & "
         f"{hc['sysser50']['lcb']['published']:.6f} & "
         f"\\num{{{hc['sysser50']['lcb']['mm']:+.2f}}}\\,mm \\\\")
out["TABGEOM"] = (r"""\begin{table}[htbp]
\centering
\caption{Recomputed hydrostatics against the published values for both hulls, at the
reference condition set by matching the published displaced volume, on a $96\times480$ mesh
per patch. The displaced volume is matched by construction. The last row is a difference in
millimetres rather than a percentage.}
\label{tab:geom}
\small
\begin{tabular}{@{}lS[table-format=1.6]cS[table-format=1.6]c@{}}
\toprule
& \multicolumn{2}{c}{Sysser 01} & \multicolumn{2}{c}{Sysser 50} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
& {published} & difference & {published} & difference \\
\midrule
""" + "\n".join(g) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# ---- Sysser 50 resistance table ----
rr = []
for k in sorted(s50):
    r = s50[k]
    m = r["rr_meas"]
    pr = f"{r['r_press'] / m:+.2f}" if abs(m) > 0.05 else "{--}"
    fr = f"{r['r_far'] / m:+.2f}" if abs(m) > 0.05 else "{--}"
    rr.append(f"{r['fn']:.3f} & {m:.3f} & {100 * r['weight_fraction']:.2f} & "
              f"{r['r_far']:.3f} & {r['r_press']:.3f} & {fr} & {pr} & "
              f"{r['resid_rms']:.3f} & {r['lambda_cap']:.2f} \\\\")
out["TABS50"] = (r"""\begin{table}[htbp]
\centering
\caption{Wave resistance of Sysser 50 by both routes at the static attitude on a
""" + str(d50["records"][0]["panels"]) + r"""-panel mesh, against the residuary resistance
reduced from the measurements, at the 13 measured speeds. Columns as in
table~\ref{tab:resistance}.}
\label{tab:s50}
\small
\begin{tabular}{@{}S[table-format=1.3]S[table-format=+2.3]S[table-format=2.2]
S[table-format=+3.3]S[table-format=+3.3]S[table-format=+3.2]S[table-format=+3.2]
S[table-format=1.3]S[table-format=2.2]@{}}
\toprule
{$\Fn$} & {$R_r$ [\si{\newton}]} & {$R_r/W$ [\si{\percent}]} &
{$R_W$ far [\si{\newton}]} & {$R_W$ pres. [\si{\newton}]} &
{far$/R_r$} & {pres.$/R_r$} & {residual} & {$\lambda_\mathrm{max}$} \\
\midrule
""" + "\n".join(rr) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# ---- attitude table ----
if piv01 or piv50:
    a = []
    for name, st, pv in (("Sysser 01", s01, piv01), ("Sysser 50", s50, piv50)):
        for k in sorted(pv):
            if k not in st:
                continue
            p, q = st[k], pv[k]
            a.append(f"{name} & {k:.2f} & {1000 * q['sinkage']:.1f} & {q['trim']:+.2f} & "
                     f"{q['volume_at_attitude'] / p['volume_at_attitude']:.3f} & "
                     f"{q['rr_meas']:.2f} & {p['r_press']:.2f} & {q['r_press']:.2f} & "
                     f"{p['r_press'] / q['rr_meas']:+.2f} & "
                     f"{q['r_press'] / q['rr_meas']:+.2f} \\\\")
    out["TABATT"] = (r"""\begin{table}[htbp]
\centering
\caption{Effect of imposing the measured running attitude, rotating about the pivot the
release determines, against running at the static attitude. The volume ratio is the
displaced volume below $z = 0$ at the running attitude over its static value, and it is the
quantity that explains the rest of the table.}
\label{tab:attitude}
\small
\begin{tabular}{@{}lS[table-format=1.2]S[table-format=2.1]S[table-format=+1.2]
S[table-format=1.3]S[table-format=2.2]S[table-format=2.2]S[table-format=2.2]
S[table-format=+2.2]S[table-format=+2.2]@{}}
\toprule
& & {sink.} & {trim} & {$\nabla/\nabla_0$} & {$R_r$} &
\multicolumn{2}{c}{$R_W$ pressure [\si{\newton}]} &
\multicolumn{2}{c}{$R_W/R_r$} \\
\cmidrule(lr){7-8}\cmidrule(lr){9-10}
hull & {$\Fn$} & {[\si{\milli\metre}]} & {[\si{\degree}]} & & {[\si{\newton}]} &
{static} & {running} & {static} & {running} \\
\midrule
""" + "\n".join(a) + r"""
\bottomrule
\end{tabular}
\end{table}""")

# ---- narrative statistics ----
st = {}
st["BTC01"] = f"\\num{{{r1['bwl'] / r1['tc']:.2f}}}"
st["BTC50"] = f"\\num{{{r50['bwl'] / r50['tc']:.2f}}}"
st["LCB01"] = f"\\num{{{-100 * r1['lcb'] / r1['lwl']:.1f}}}"
st["LCB50"] = f"\\num{{{-100 * r50['lcb'] / r50['lwl']:.1f}}}"
if 0.3 in s50 and 0.3 in s01:
    st["OSC50"] = f"\\num{{{s50[0.3]['osc_worst']:.0f}}}"
    st["OSC01"] = f"\\num{{{s01[0.3]['osc_worst']:.0f}}}"
    st["OSCCOMPARE"] = (f"the worst count is \\num{{{s50[0.3]['osc_worst']:.0f}}} on the "
                        f"Sysser 50 mesh against \\num{{{s01[0.3]['osc_worst']:.0f}}} on the "
                        f"Sysser 01 mesh at $\\Fn = 0.30$, with no pair outside the envelope "
                        f"on either")
if piv01 or piv50:
    ratios = [q["volume_at_attitude"] / s01[k]["volume_at_attitude"]
              for k, q in piv01.items() if k in s01]
    ratios += [q["volume_at_attitude"] / s50[k]["volume_at_attitude"]
               for k, q in piv50.items() if k in s50]
    st["VOLGAIN"] = (f"\\SIrange{{{100 * (min(ratios) - 1):.0f}}}"
                     f"{{{100 * (max(ratios) - 1):.0f}}}{{\\percent}} above static")
# ---- attitude narrative ----
pairs = [(k, s01[k], piv01[k]) for k in sorted(piv01) if k in s01]
if pairs:
    rise = [q["r_press"] / p["r_press"] for _, p, q in pairs]
    vol = [q["volume_at_attitude"] / p["volume_at_attitude"] for _, p, q in pairs]
    k_list = ", ".join(f"{k:.2f}" for k, _, _ in pairs)
    st["ATTITUDERESULT"] = (
        "On Sysser 01, at $\\Fn = " + k_list + "$, the pressure route rises by "
        + ", ".join(f"\\SI{{{100 * (r - 1):.0f}}}{{\\percent}}" for r in rise)
        + " while the displaced volume below $z = 0$ rises by "
        + ", ".join(f"\\SI{{{100 * (v - 1):.0f}}}{{\\percent}}" for v in vol)
        + ". The resistance rise divided by the volume rise is "
        + ", ".join(f"\\num{{{r / v:.2f}}}" for r, v in zip(rise, vol))
        + ", so one quantity accounts for the effect to within "
        + f"\\SI{{{100 * (max(r / v for r, v in zip(rise, vol)) / min(r / v for r, v in zip(rise, vol)) - 1):.0f}}}{{\\percent}}"
        + " across the range.")
    st["VOLRISE"] = ("\\SIrange{%.0f}{%.0f}{\\percent}"
                     % (100 * (min(rise) - 1), 100 * (max(rise) - 1)))
    lo = [p["r_press"] / q["rr_meas"] for k, p, q in pairs if k >= 0.45]
    hi = [q["r_press"] / q["rr_meas"] for k, p, q in pairs if k >= 0.45]
    if lo:
        ks = [k for k, _, _ in pairs if k >= 0.45]
        st["BRACKET01"] = (f"\\numrange{{{min(lo):.2f}}}{{{max(hi):.2f}}} of measurement "
                           f"over $\\Fn = {min(ks):.2f}$ to ${max(ks):.2f}$")
    top = max(pairs, key=lambda t: t[0])
    st["FIXRATIO"] = (f"\\num{{{top[1]['r_press'] / top[2]['rr_meas']:.2f}}} to "
                      f"\\num{{{top[2]['r_press'] / top[2]['rr_meas']:.2f}}} of measurement "
                      f"at $\\Fn = {top[0]:.2f}$")

# ---- friction sensitivity of the reduction ----
if 0.3 in s01:
    r = s01[0.3]
    st["FRICSHIFT"] = ("\\SI{%.0f}{\\percent}"
                       % abs(100 * 0.021 * r["rf_meas"] / r["rr_meas"]))

# ---- beam / discretisation narrative, and the two-hull comparison table ----
common = sorted(set(s01) & set(s50))
hi = [k for k in common if k >= 0.25]
lo = [k for k in common if k < 0.25]
if hi and lo:
    def stats(d, ks):
        res = [d[k]["resid_rms"] for k in ks]
        rat = [d[k]["r_press"] / d[k]["rr_meas"] for k in ks if abs(d[k]["rr_meas"]) > 0.05]
        fp = [d[k]["r_far"] / d[k]["r_press"] for k in ks if d[k]["r_press"] > 0]
        return res, rat, fp

    r1h, a1h, f1h = stats(s01, hi)
    r5h, a5h, f5h = stats(s50, hi)
    r1l, _, _ = stats(s01, lo)
    r5l, _, _ = stats(s50, lo)
    mean = lambda v: sum(v) / len(v)
    st["RESLO50"] = f"\\numrange{{{min(r5l):.3f}}}{{{max(r5l):.3f}}}"
    st["RESLO01"] = f"\\numrange{{{min(r1l):.3f}}}{{{max(r1l):.3f}}}"
    st["RESHI"] = (
        f"the mean residual is \\num{{{mean(r5h):.3f}}} of $u$ on Sysser 50 against "
        f"\\num{{{mean(r1h):.3f}}} on Sysser 01, the ratio of prediction to measurement "
        f"spans \\numrange{{{min(a5h):.2f}}}{{{max(a5h):.2f}}} against "
        f"\\numrange{{{min(a1h):+.2f}}}{{{max(a1h):.2f}}}, and the two resistance routes "
        f"differ by at most a factor of \\num{{{max(f5h):.2f}}} against "
        f"\\num{{{max(f1h):.2f}}}")
    rows = []
    for lab, d, ks in (("Sysser 01", s01, hi), ("Sysser 50", s50, hi)):
        res, rat, fp = stats(d, ks)
        rl = stats(d, lo)[0]
        rows.append(f"{lab} & {d[ks[0]]['panels']} & "
                    f"{min(rl):.3f}--{max(rl):.3f} & {mean(res):.3f} & {max(res):.3f} & "
                    f"{min(rat):+.2f}--{max(rat):+.2f} & {min(fp):.2f}--{max(fp):.2f} \\\\")
    out["TABCOMPARE"] = (r"""\begin{table}[htbp]
\centering
\caption{The two hulls compared at the static attitude on the diagnostics that measure
discretisation error, split at $\Fn = 0.25$. The residual is the root-mean-square
body-condition residual at points that are not collocation points, as a fraction of $u$; the
last column is the ratio of the far-field route to the pressure route, which is unity for a
converged solution.}
\label{tab:compare}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
& & residual, & \multicolumn{2}{c}{residual, $\Fn \ge 0.25$}
& \multicolumn{2}{c}{$\Fn \ge 0.25$} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
hull & $N$ & $\Fn < 0.25$ & mean & max & $R_W/R_r$ & far\,/\,pressure \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")
    top = [k for k in s50 if k >= 0.45]
    fp50 = [abs(s50[k]["r_far"] / s50[k]["r_press"] - 1.0) for k in top]
    ra50 = [s50[k]["r_press"] / s50[k]["rr_meas"] for k in top]
    st["ROUTEAGREE"] = f"\\SI{{{100 * max(fp50):.0f}}}{{\\percent}} or better"
    st["ROUTERATIO"] = f"\\numrange{{{min(ra50):.2f}}}{{{max(ra50):.2f}}}"

if hi:
    a1 = [s01[k]["r_press"] / s01[k]["rr_meas"] for k in s01 if abs(s01[k]["rr_meas"]) > 0.05]
    a5 = [s50[k]["r_press"] / s50[k]["rr_meas"] for k in s50 if abs(s50[k]["rr_meas"]) > 0.05]
    st["RAT01"] = f"\\numrange{{{min(a1):+.2f}}}{{{max(a1):.2f}}}"
    st["RAT50"] = f"\\numrange{{{min(a5):.2f}}}{{{max(a5):.2f}}}"

json.dump(out | st, open(B / "extra50.json", "w"), indent=1)
print("extra50 keys:", sorted(out) , "+", sorted(st))
