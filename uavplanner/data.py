"""Synthetic example data and CSV reading/writing.

CSV format (comma-separated, first row = column names):

    id,x,y,profit,service_time
    0,15000,15000,0,0          <- optional row: the docking station (id 0)
    1,2300.5,18000,4,120
    2,9100,4200,1,120

* x and y are in metres on a flat map.
* profit is the urgency of the site (for example 1 to 4).
* service_time (seconds) is optional. If it is missing, the default is used.
* id is optional. If no row has id 0, the docking station is placed at the
  centre of all the sites.
"""

from __future__ import annotations

import csv
import io
import random
from typing import Dict, Optional

from .model import Instance, Site

# The mix of profits in the paper's real case (88 sites, total profit 200).
PAPER_PROFIT_MIX = {4: 15, 3: 21, 2: 25, 1: 27}


def generate_synthetic(
    seed: int = 1,
    region_km: float = 30.0,
    service_time: float = 120.0,
    profit_mix: Optional[Dict[int, int]] = None,
    t_max: float = 7200.0,
    rho: float = 0.9,
    speed: float = 16.0,
) -> Instance:
    """Made-up example: random site locations in a square region, with the
    docking station in the middle. Same seed = same example every time."""
    rng = random.Random(seed)
    mix = profit_mix if profit_mix is not None else PAPER_PROFIT_MIX
    profits = []
    for value, count in mix.items():
        profits += [float(value)] * count
    rng.shuffle(profits)

    side = region_km * 1000.0
    depot = (side / 2.0, side / 2.0)
    sites = [
        Site(
            id=k,
            x=rng.uniform(0.0, side),
            y=rng.uniform(0.0, side),
            profit=p,
            service_time=service_time,
        )
        for k, p in enumerate(profits, start=1)
    ]
    return Instance(depot, sites, t_max=t_max, rho=rho, speed=speed)


def _cell(row, names, column):
    key = names.get(column)
    if key is None:
        return None
    value = row.get(key)
    return value.strip() if value is not None else None


def parse_csv_text(
    text: str,
    t_max: float = 7200.0,
    rho: float = 0.9,
    speed: float = 16.0,
    default_service_time: float = 120.0,
) -> Instance:
    """Build an Instance from the text of a CSV file."""
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("The CSV file is empty.")

    first_line = text.splitlines()[0]
    delimiter = ";" if (";" in first_line and "," not in first_line) else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError("The CSV file has no header row.")

    names = {name.strip().lower(): name for name in reader.fieldnames if name}
    for required in ("x", "y", "profit"):
        if required not in names:
            raise ValueError(
                f"The CSV file needs a column called '{required}'. "
                f"Columns found: {', '.join(names)}."
            )

    depot = None
    rows = []
    for line_no, row in enumerate(reader, start=2):
        if all((v is None or str(v).strip() == "") for v in row.values()):
            continue  # skip blank lines
        try:
            x = float(_cell(row, names, "x"))
            y = float(_cell(row, names, "y"))
            profit = float(_cell(row, names, "profit"))
            service_raw = _cell(row, names, "service_time")
            service = float(service_raw) if service_raw else default_service_time
        except (TypeError, ValueError):
            raise ValueError(
                f"Line {line_no} of the CSV file has a missing or non-numeric value "
                f"in x, y, profit or service_time."
            )
        id_raw = _cell(row, names, "id")
        is_depot = False
        if id_raw:
            try:
                is_depot = int(float(id_raw)) == 0
            except ValueError:
                is_depot = False
        if is_depot:
            depot = (x, y)
        else:
            rows.append((x, y, profit, service))

    if not rows:
        raise ValueError("The CSV file contains no sites.")
    if depot is None:
        depot = (sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows))

    sites = [Site(k, x, y, p, s) for k, (x, y, p, s) in enumerate(rows, start=1)]
    return Instance(depot, sites, t_max=t_max, rho=rho, speed=speed)


def load_csv(path: str, **kwargs) -> Instance:
    """Read a CSV file from disk (see parse_csv_text for options)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return parse_csv_text(f.read(), **kwargs)


def instance_to_csv_text(instance: Instance) -> str:
    """Write an instance as CSV text (docking station = row with id 0)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["id", "x", "y", "profit", "service_time"])
    writer.writerow([0, round(instance.depot[0], 2), round(instance.depot[1], 2), 0, 0])
    for s in instance.sites:
        writer.writerow([s.id, round(s.x, 2), round(s.y, 2), s.profit, s.service_time])
    return out.getvalue()