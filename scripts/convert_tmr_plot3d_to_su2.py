from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_plot3d_2d(path: Path) -> tuple[int, int, list[float], list[float]]:
    tokens = path.read_text(encoding="utf-8").split()
    pos = 0
    nblocks = int(tokens[pos])
    pos += 1
    if nblocks != 1:
        raise ValueError(f"Expected one PLOT3D block, found {nblocks}")
    ni = int(tokens[pos])
    nj = int(tokens[pos + 1])
    pos += 2
    npts = ni * nj
    x = [float(v) for v in tokens[pos : pos + npts]]
    pos += npts
    y = [float(v) for v in tokens[pos : pos + npts]]
    if len(x) != npts or len(y) != npts:
        raise ValueError("Unexpected PLOT3D coordinate count")
    return ni, nj, x, y


def pidx(i: int, j: int, ni: int) -> int:
    return j * ni + i


def read_nmf_airfoil_range(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"'viscous_solid'\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)", text)
    if not match:
        raise ValueError(f"Could not find viscous_solid range in {path}")
    start = int(match.group(1))
    end = int(match.group(2))
    return start, end


def build_index_map(
    ni: int, nj: int, airfoil_start_1based: int, airfoil_end_1based: int
) -> tuple[dict[tuple[int, int], int], list[tuple[tuple[int, int], int]]]:
    # TMR C-grid wake duplicate points lie on the inner boundary. The neutral map
    # pairs i=1..wake_count with i=ni..airfoil_end, reversed. This matches the
    # SU2 tutorial mesh point count: 897*257 - 193 = 230336.
    wake_count = airfoil_start_1based
    duplicate_start = airfoil_end_1based - 1
    mapping: dict[tuple[int, int], int] = {}
    points: list[tuple[tuple[int, int], int]] = []

    next_id = 0
    for j in range(nj):
        for i in range(ni):
            key = (i, j)
            if j == 0 and i >= duplicate_start:
                mirrored_i = ni - 1 - i
                mapping[key] = mapping[(mirrored_i, j)]
                continue
            mapping[key] = next_id
            points.append((key, pidx(i, j, ni)))
            next_id += 1
    return mapping, points


def write_su2(p2d_path: Path, nmf_path: Path, out_path: Path) -> None:
    ni, nj, x, y = read_plot3d_2d(p2d_path)
    airfoil_start_1based, airfoil_end_1based = read_nmf_airfoil_range(nmf_path)
    mapping, points = build_index_map(ni, nj, airfoil_start_1based, airfoil_end_1based)

    nelem = (ni - 1) * (nj - 1)
    airfoil_start = airfoil_start_1based - 1
    airfoil_end = airfoil_end_1based - 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("NDIME=2\n")
        f.write(f"NELEM={nelem}\n")
        elem_id = 0
        for j in range(nj - 1):
            for i in range(ni - 1):
                n0 = mapping[(i, j)]
                n1 = mapping[(i + 1, j)]
                n2 = mapping[(i + 1, j + 1)]
                n3 = mapping[(i, j + 1)]
                f.write(f"9 \t {n0} \t {n1} \t {n2} \t {n3} \t {elem_id}\n")
                elem_id += 1

        f.write(f"NPOIN={len(points)}\n")
        for new_id, ((_, _), old_id) in enumerate(points):
            f.write(f"{x[old_id]:.15E} \t {y[old_id]:.15E} \t {new_id}\n")

        f.write("NMARK=2\n")
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS={airfoil_end - airfoil_start}\n")
        for i in range(airfoil_start, airfoil_end):
            f.write(f"3 \t {mapping[(i, 0)]} \t {mapping[(i + 1, 0)]}\n")

        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS={(ni - 1) + 2 * (nj - 1)}\n")
        # Outer boundary.
        for i in range(ni - 1):
            f.write(f"3 \t {mapping[(i, nj - 1)]} \t {mapping[(i + 1, nj - 1)]}\n")
        # Lower and upper wake/farfield cuts.
        for j in range(nj - 1):
            f.write(f"3 \t {mapping[(0, j)]} \t {mapping[(0, j + 1)]}\n")
        for j in range(nj - 1):
            f.write(f"3 \t {mapping[(ni - 1, j)]} \t {mapping[(ni - 1, j + 1)]}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("p2d", type=Path)
    parser.add_argument("nmf", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    write_su2(args.p2d, args.nmf, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
