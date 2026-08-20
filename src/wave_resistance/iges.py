"""Minimal, strict IGES reader.

Scope for this milestone: entity type 128 (rational B-spline surface), plus type 124
(transformation matrix) because a silently ignored transformation would corrupt the
geometry.  Every other entity is skipped and reported by name, never guessed at.

Reference: IGES 5.3 specification.  Lines are 80 characters; column 73 (1-based)
carries the section letter S/G/D/P/T.  Parameter-data payload lives in columns 1-64
and the owning directory-entry sequence number in columns 65-72.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["NurbsSurface", "IgesFile", "read_iges", "UNIT_SCALE_TO_METRE"]

# IGES global field 14: units flag -> metres per file unit.
UNIT_SCALE_TO_METRE = {
    1: 0.0254,      # inch
    2: 1.0e-3,      # millimetre
    4: 0.3048,      # foot
    5: 1609.344,    # mile
    6: 1.0,         # metre
    7: 1000.0,      # kilometre
    8: 2.54e-5,     # mil
    9: 1.0e-6,      # micron
    10: 1.0e-2,     # centimetre
    11: 2.54e-8,    # microinch
}

_HOLLERITH = re.compile(r"^(\d+)H")
_ENTITY_NAMES = {
    100: "circular arc", 102: "composite curve", 104: "conic arc", 106: "copious data",
    108: "plane", 110: "line", 112: "parametric spline curve", 114: "parametric spline surface",
    116: "point", 118: "ruled surface", 120: "surface of revolution", 122: "tabulated cylinder",
    124: "transformation matrix", 126: "rational B-spline curve", 128: "rational B-spline surface",
    140: "offset surface", 141: "boundary", 142: "curve on a parametric surface",
    143: "bounded surface", 144: "trimmed parametric surface", 186: "manifold solid B-rep",
    202: "linear dimension", 212: "general note", 308: "subfigure definition",
    314: "colour definition", 402: "associativity instance", 406: "property",
    408: "singular subfigure instance", 410: "view", 502: "vertex list", 504: "edge list",
    508: "loop", 510: "face", 514: "shell",
}


def _to_float(token: str) -> float:
    """IGES writes doubles as 1.0D0; Python needs 1.0E0."""
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _tokenise(payload: str, param_delim: str, record_delim: str) -> list[str]:
    """Split a free-format IGES section, honouring nnH Hollerith strings.

    A Hollerith string may contain the delimiter, so it must be consumed by length
    rather than by splitting.  Empty fields are preserved as empty strings because
    IGES uses them to mean "take the default".
    """
    out: list[str] = []
    i, n = 0, len(payload)
    current = ""
    while i < n:
        ch = payload[i]
        if ch == record_delim:
            out.append(current)
            return out
        if ch == param_delim:
            out.append(current)
            current = ""
            i += 1
            continue
        m = _HOLLERITH.match(payload[i:])
        if m and not current.strip():
            length = int(m.group(1))
            start = i + m.end()
            out.append(payload[start:start + length])
            i = start + length
            current = ""
            # Consume the delimiter that must follow the string.
            if i < n and payload[i] == param_delim:
                i += 1
            elif i < n and payload[i] == record_delim:
                return out
            continue
        current += ch
        i += 1
    out.append(current)
    return out


@dataclass(frozen=True)
class NurbsSurface:
    """A tensor-product NURBS surface patch, with coordinates already in metres.

    ``control_points`` has shape ``(n_u, n_v, 3)`` and ``weights`` shape ``(n_u, n_v)``,
    i.e. the first axis is the u direction.  IGES stores the u index varying fastest;
    the reader has already undone that.
    """

    degree_u: int
    degree_v: int
    knots_u: np.ndarray
    knots_v: np.ndarray
    control_points: np.ndarray
    weights: np.ndarray
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    de_pointer: int
    form: int
    label: str = ""
    closed_u: bool = False
    closed_v: bool = False
    is_polynomial: bool = True

    def __post_init__(self) -> None:
        n_u, n_v = self.control_points.shape[:2]
        if self.control_points.shape != (n_u, n_v, 3):
            raise ValueError(f"control_points must be (n_u, n_v, 3), got {self.control_points.shape}")
        if self.weights.shape != (n_u, n_v):
            raise ValueError(f"weights must be (n_u, n_v), got {self.weights.shape}")
        if self.knots_u.size != n_u + self.degree_u + 1:
            raise ValueError(
                f"u knot count {self.knots_u.size} != n_u + degree_u + 1 "
                f"= {n_u + self.degree_u + 1}"
            )
        if self.knots_v.size != n_v + self.degree_v + 1:
            raise ValueError(
                f"v knot count {self.knots_v.size} != n_v + degree_v + 1 "
                f"= {n_v + self.degree_v + 1}"
            )
        if np.any(np.diff(self.knots_u) < -1e-12) or np.any(np.diff(self.knots_v) < -1e-12):
            raise ValueError("knot vectors must be non-decreasing")
        if np.any(self.weights <= 0.0):
            raise ValueError("NURBS weights must be strictly positive")

    @property
    def n_u(self) -> int:
        return self.control_points.shape[0]

    @property
    def n_v(self) -> int:
        return self.control_points.shape[1]

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Control-net bounding box: a true bound on the surface by the convex-hull property."""
        pts = self.control_points.reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)


@dataclass
class IgesFile:
    surfaces: list[NurbsSurface] = field(default_factory=list)
    units_flag: int = 6
    units_name: str = "M"
    metres_per_unit: float = 1.0
    model_space_scale: float = 1.0
    ignored: Counter = field(default_factory=Counter)
    source: Path | None = None

    def report(self) -> str:
        lines = [
            f"IGES file: {self.source}",
            f"  units: flag {self.units_flag} ({self.units_name}) "
            f"-> {self.metres_per_unit:g} m per file unit",
            f"  model space scale: {self.model_space_scale:g}",
            f"  type 128 surfaces read: {len(self.surfaces)}",
        ]
        for s in self.surfaces:
            lo, hi = s.bounding_box()
            lines.append(
                f"    DE {s.de_pointer:>4d}  {s.n_u}x{s.n_v} cps  degree {s.degree_u}x{s.degree_v}"
                f"  form {s.form}  {'polynomial' if s.is_polynomial else 'rational'}"
                f"  label {s.label!r}"
            )
            lines.append(
                f"           bbox [{lo[0]:.4f},{hi[0]:.4f}] x [{lo[1]:.4f},{hi[1]:.4f}]"
                f" x [{lo[2]:.4f},{hi[2]:.4f}] m"
            )
        if self.ignored:
            lines.append("  ignored entities:")
            for etype, count in sorted(self.ignored.items()):
                lines.append(f"    type {etype} ({_ENTITY_NAMES.get(etype, 'unknown')}): {count}")
        return "\n".join(lines)


def _parse_global(lines: list[str]) -> tuple[list[str], str, str]:
    payload = "".join(l[:72].rstrip() for l in lines)
    # Tokenise once with the defaults to discover the actual delimiters, then again
    # with those if they differ.
    fields = _tokenise(payload, ",", ";")
    param_delim = fields[0] if fields and fields[0] else ","
    record_delim = fields[1] if len(fields) > 1 and fields[1] else ";"
    if (param_delim, record_delim) != (",", ";"):
        fields = _tokenise(payload, param_delim, record_delim)
    return fields, param_delim, record_delim


def _parse_128(values: list[str], de: int, form: int, label: str, scale: float) -> NurbsSurface:
    k1, k2, m1, m2 = (int(values[i]) for i in (1, 2, 3, 4))
    closed_u, closed_v = bool(int(values[5])), bool(int(values[6]))
    polynomial = bool(int(values[7]))
    n_u, n_v = k1 + 1, k2 + 1
    n_knots_u, n_knots_v = k1 + m1 + 2, k2 + m2 + 2
    n_w = n_u * n_v
    need = 10 + n_knots_u + n_knots_v + n_w + 3 * n_w + 4
    if len(values) < need:
        raise ValueError(
            f"entity 128 at DE {de}: expected at least {need} parameter fields, got {len(values)}"
        )
    cur = 10
    knots_u = np.array([_to_float(v) for v in values[cur:cur + n_knots_u]]); cur += n_knots_u
    knots_v = np.array([_to_float(v) for v in values[cur:cur + n_knots_v]]); cur += n_knots_v
    weights = np.array([_to_float(v) for v in values[cur:cur + n_w]]); cur += n_w
    xyz = np.array([_to_float(v) for v in values[cur:cur + 3 * n_w]]); cur += 3 * n_w
    u0, u1, v0, v1 = (_to_float(v) for v in values[cur:cur + 4])
    # IGES orders these with the u index varying fastest.
    weights = weights.reshape(n_v, n_u).T.copy()
    control = xyz.reshape(n_v, n_u, 3).transpose(1, 0, 2).copy() * scale
    return NurbsSurface(
        degree_u=m1, degree_v=m2, knots_u=knots_u, knots_v=knots_v,
        control_points=control, weights=weights,
        u_range=(u0, u1), v_range=(v0, v1),
        de_pointer=de, form=form, label=label.strip(),
        closed_u=closed_u, closed_v=closed_v, is_polynomial=polynomial,
    )


def read_iges(path: str | Path) -> IgesFile:
    """Read the type 128 surfaces from an IGES file, converting lengths to metres."""
    path = Path(path)
    raw = [l.rstrip("\n").rstrip("\r") for l in path.open("r", encoding="latin-1")]
    sections: dict[str, list[str]] = {k: [] for k in "SGDPT"}
    for line in raw:
        if len(line) < 73:
            continue
        letter = line[72]
        if letter in sections:
            sections[letter].append(line)
    if not sections["D"] or not sections["P"]:
        raise ValueError(f"{path}: no directory or parameter section found")

    gfields, param_delim, record_delim = _parse_global(sections["G"])

    def gfield(idx: int, default: str = "") -> str:
        return gfields[idx - 1] if len(gfields) >= idx and gfields[idx - 1] != "" else default

    model_scale = _to_float(gfield(13, "1.0"))
    units_flag = int(gfield(14, "6"))
    units_name = gfield(15, "M")
    if units_flag not in UNIT_SCALE_TO_METRE:
        raise ValueError(
            f"{path}: units flag {units_flag} ({units_name!r}) is not supported; "
            "supported flags are " + ", ".join(map(str, sorted(UNIT_SCALE_TO_METRE)))
        )
    metres_per_unit = UNIT_SCALE_TO_METRE[units_flag]
    scale = metres_per_unit * model_scale

    # Directory entries: two 80-char lines per entity, nine 8-char fields per line.
    dlines = sections["D"]
    if len(dlines) % 2:
        raise ValueError(f"{path}: directory section has an odd number of lines")
    directory: dict[int, dict] = {}
    for k in range(0, len(dlines), 2):
        a, b = dlines[k], dlines[k + 1]
        f = [a[i:i + 8] for i in range(0, 72, 8)] + [b[i:i + 8] for i in range(0, 72, 8)]
        de_seq = int(a[73:80])
        directory[de_seq] = {
            "type": int(f[0]),
            "pd_pointer": int(f[1]),
            "transform": int(f[6]) if f[6].strip() else 0,
            "line_count": int(f[12]) if f[12].strip() else 0,
            "form": int(f[13]) if f[13].strip() else 0,
            "label": f[16],
        }

    # Parameter data grouped by the owning DE sequence number in columns 65-72.
    groups: "OrderedDict[int, list[str]]" = OrderedDict()
    for line in sections["P"]:
        ptr = line[64:72].strip()
        if not ptr:
            continue
        groups.setdefault(int(ptr), []).append(line[:64])

    result = IgesFile(
        units_flag=units_flag, units_name=units_name, metres_per_unit=metres_per_unit,
        model_space_scale=model_scale, source=path,
    )

    # Transformation matrices first, so a surface can reference one.
    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for de, meta in directory.items():
        if meta["type"] != 124:
            continue
        vals = _tokenise("".join(groups.get(de, [])), param_delim, record_delim)
        nums = [_to_float(v) for v in vals[1:13]]
        rot = np.array([nums[0:3], nums[4:7], nums[8:11]])
        trans = np.array([nums[3], nums[7], nums[11]]) * scale
        transforms[de] = (rot, trans)

    for de, meta in sorted(directory.items()):
        etype = meta["type"]
        if etype == 124:
            continue
        if etype != 128:
            result.ignored[etype] += 1
            continue
        payload = "".join(groups.get(de, []))
        if not payload:
            raise ValueError(f"{path}: entity 128 at DE {de} has no parameter data")
        values = _tokenise(payload, param_delim, record_delim)
        surface = _parse_128(values, de, meta["form"], meta["label"], scale)
        tptr = meta["transform"]
        if tptr:
            if tptr not in transforms:
                raise ValueError(
                    f"{path}: entity 128 at DE {de} references transformation matrix "
                    f"DE {tptr}, which was not found"
                )
            rot, trans = transforms[tptr]
            moved = surface.control_points.reshape(-1, 3) @ rot.T + trans
            surface = NurbsSurface(
                degree_u=surface.degree_u, degree_v=surface.degree_v,
                knots_u=surface.knots_u, knots_v=surface.knots_v,
                control_points=moved.reshape(surface.control_points.shape),
                weights=surface.weights, u_range=surface.u_range, v_range=surface.v_range,
                de_pointer=de, form=surface.form, label=surface.label,
                closed_u=surface.closed_u, closed_v=surface.closed_v,
                is_polynomial=surface.is_polynomial,
            )
        result.surfaces.append(surface)

    if not result.surfaces:
        raise ValueError(f"{path}: no type 128 surfaces found")
    return result
