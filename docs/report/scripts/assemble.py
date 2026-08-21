"""Fill the report template with the measured numbers and the section text."""
import json, pathlib, subprocess, datetime

B = pathlib.Path("/tmp/claude-0/-home-user-wave-resistance/af901ab6-6add-5d9e-93ac-9e92272f9553/scratchpad")
P = B / "parts"
repo = pathlib.Path("/home/user/wave-resistance")

ver = json.load(open(B / "verify.json"))
ver2 = json.load(open(B / "verify2.json"))
env = json.load(open(B / "envelope.json"))
hyd = json.load(open(B / "hydro.json"))
st = json.load(open(B / "s280_static.json"))
rec = [r for r in st["records"] if r.get("panels")]
by_fn = {round(r["fn"], 2): r for r in rec}
try:
    me = json.load(open(B / "s280_meas.json"))
    mrec = [r for r in me["records"] if r.get("panels")]
except Exception:
    mrec = []
try:
    f480 = json.load(open(B / "s480_static.json"))
    r480 = {round(r["fn"], 2): r for r in f480["records"] if r.get("panels")}
except Exception:
    r480 = {}

sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()


def sci(v, d=1):
    """LaTeX \num-friendly scientific form."""
    return f"\\num{{{v:.{d}e}}}"


# ---------- hydrostatics table ----------
pub = ver["hydro_vs_published"]
order = [("volume", r"$\nabla_c$", r"\si{\metre\cubed}", 6),
         ("waterplane_area", r"$A_w$", r"\si{\metre\squared}", 6),
         ("wetted_area", r"$S_c$", r"\si{\metre\squared}", 6),
         ("lwl", r"$c$", r"\si{\metre}", 6),
         ("bwl", r"$b$", r"\si{\metre}", 6),
         ("tc", r"$d$", r"\si{\metre}", 6),
         ("cp", r"$C_\mathrm{p}$", "---", 6),
         ("cb", r"$C_\mathrm{b}$", "---", 6),
         ("cwp", r"$C_\mathrm{wp}$", "---", 6)]
rows = []
for key, sym, unit, d in order:
    e = pub[key]
    rows.append(f"{sym} & {unit} & {e['computed']:.{d}f} & {e['published']:.{d}f} & "
                f"{e['pct']:+.3f} \\\\")
tabhydro = r"""\begin{table}[htbp]
\centering
\caption{Hydrostatics of Sysser 01 recomputed at the reference condition set by matching
the published displaced volume, against the values published in the DSYHS hydrostatics
release, on a $48\times240$ mesh per patch.}
\label{tab:hydro}
\small
\begin{tabular}{@{}llS[table-format=1.6]S[table-format=1.6]S[table-format=+1.3]@{}}
\toprule
& unit & {computed} & {published} & {difference [\si{\percent}]} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

# ---------- resistance table ----------
res_rows = []
for r in sorted(rec, key=lambda r: r["fn"]):
    rr = r["rr_meas"]
    ratio = f"{r['r_press'] / rr:+.2f}" if abs(rr) > 0.05 else "{--}"
    ratio_f = f"{r['r_far'] / rr:+.2f}" if abs(rr) > 0.05 else "{--}"
    res_rows.append(
        f"{r['fn']:.3f} & {rr:.3f} & {100 * r['weight_fraction']:.2f} & "
        f"{r['r_far']:.3f} & {r['r_press']:.3f} & {ratio_f} & {ratio} & "
        f"{r['resid_rms']:.3f} & {r['lambda_cap']:.2f} \\\\")
tabres = r"""\begin{table}[htbp]
\centering
\caption{Wave resistance of Sysser 01 by both routes at the static attitude on the
280-panel mesh, against the residuary resistance reduced from the measurements, for Froude
numbers from 0.10 to 0.60. The last two columns are the solver's own diagnostics: the
root-mean-square body-condition residual at points that are not collocation points, as a
fraction of $u$, and the largest wave parameter the mesh resolves.}
\label{tab:resistance}
\small
\begin{tabular}{@{}S[table-format=1.3]S[table-format=+2.3]S[table-format=1.2]
S[table-format=+2.3]S[table-format=+2.3]S[table-format=+2.2]S[table-format=+2.2]
S[table-format=1.3]S[table-format=2.2]@{}}
\toprule
{$\Fn$} & {$R_r$ [\si{\newton}]} & {$R_r/W$ [\si{\percent}]} &
{$R_W$ far [\si{\newton}]} & {$R_W$ pres. [\si{\newton}]} &
{far$/R_r$} & {pres.$/R_r$} & {residual} & {$\lambda_\mathrm{max}$} \\
\midrule
""" + "\n".join(res_rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

subs = {
 "COMMITSHA": sha,
 "REPORTDATE": datetime.date.today().strftime("%d %B %Y"),
 "VOLSPREAD": sci(ver["volume_route_rel_spread"]),
 "VOLRATIO": f"{ver['volume_ratio']:.2f}",
 "VOLRICH": f"{ver['volume_richardson']:.7f}",
 "FSMED": sci(ver2["free_surface_residual"]["median"]),
 "FSWORST": sci(ver2["free_surface_residual"]["worst"]),
 "GRADMED": sci(ver2["gradient_vs_fd"]["median"]),
 "GRADWORST": sci(ver2["gradient_vs_fd"]["worst"]),
 "ENV30W": sci(env["fn0.30"]["worst"]),
 "ENV30M": sci(env["fn0.30"]["median"]),
 "ENV45W": sci(env["fn0.45"]["worst"]),
 "CAP010": f"{ver['lambda_cap_by_fn']['0.10']:.2f}",
 "TABHYDRO": tabhydro,
 "TABRES": tabres,
}
json.dump(subs, open(B / "subs.json", "w"), indent=1, default=str)
print("substitutions ready:", len(subs))

# ---------- assemble the document ----------
tex = (repo / "docs/report/report.tex").read_text()
part = lambda n: (P / n).read_text().strip()
tex = tex.replace("ABSTRACTBODY", part("abstract.tex"))
tex = tex.replace("INTRO", part("intro.tex"))
tex = tex.replace("METHOD", part("method.tex"))
tex = tex.replace("VERIF", part("verif.tex"))
tex = tex.replace("RESULTS", part("results.tex"))
tex = tex.replace("CAPLIM", part("caplim.tex"))
tex = tex.replace("CONCL", part("concl.tex"))
for k, v in subs.items():
    tex = tex.replace(k, str(v))
(repo / "docs/report/wave-resistance-report.tex").write_text(tex)
print("built", len(tex), "chars; unresolved:",
      [k for k in ("RESULTNARRATIVE", "RESABSTRACT", "CONCLRESULT") if k in tex])

# ---------- convergence and Michell tables, appended after the main build ----------
extra = {}
if r480:
    lines = []
    for k in sorted(r480):
        a = by_fn.get(k)
        b = r480[k]
        if not a:
            continue
        lines.append(
            f"{k:.2f} & {a['r_far']:.3f} & {b['r_far']:.3f} & "
            f"{100 * (b['r_far'] / a['r_far'] - 1):+.1f} & "
            f"{a['r_press']:.3f} & {b['r_press']:.3f} & "
            f"{100 * (b['r_press'] / a['r_press'] - 1):+.1f} & "
            f"{a['resid_rms']:.3f} & {b['resid_rms']:.3f} \\\\")
    extra["TABCONV"] = (r"""\begin{table}[htbp]
\centering
\caption{Effect of refining the mesh from 280 to 480 panels on the wave resistance of
Sysser 01 at the static attitude, by both routes, with the root-mean-square
body-condition residual measured away from the collocation points.}
\label{tab:convergence}
\small
\begin{tabular}{@{}S[table-format=1.2]S[table-format=2.3]S[table-format=2.3]
S[table-format=+3.1]S[table-format=1.3]S[table-format=1.3]S[table-format=+3.1]
S[table-format=1.3]S[table-format=1.3]@{}}
\toprule
& \multicolumn{3}{c}{$R_W$ far field [\si{\newton}]}
& \multicolumn{3}{c}{$R_W$ pressure [\si{\newton}]}
& \multicolumn{2}{c}{residual} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
{$\Fn$} & {$N=280$} & {$N=480$} & {change [\si{\percent}]}
& {$N=280$} & {$N=480$} & {change [\si{\percent}]}
& {$N=280$} & {$N=480$} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}""")
json.dump(extra, open(B / "extra.json", "w"), indent=1)
print("extra tables:", list(extra))
