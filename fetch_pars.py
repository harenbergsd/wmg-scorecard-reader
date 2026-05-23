#!/usr/bin/env python3
"""Fetch par data from mywmgt.com and write data/pars.csv.

Usage:
    python fetch_pars.py
    python fetch_pars.py --output path/to/pars.csv

The script pages through the APEX Interactive Report at:
    https://mywmgt.com/ords/r/fhit/wmgt/strokes

Hard-course names are derived from the paired easy-course name
(e.g. code 20H → "20,000 Leagues Under The Sea - Hard").
"""

import argparse
import csv
import re
import sys

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://mywmgt.com/ords/r/fhit/wmgt/strokes"
AJAX_URL = "https://mywmgt.com/ords/wwv_flow.ajax"
DEFAULT_OUTPUT = "data/pars.csv"
HOLE_NUMBERS = list(range(1, 19))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Page state extraction
# ---------------------------------------------------------------------------


def _extract_state(soup, html):
    """Extract APEX session/widget state needed for AJAX pagination calls."""
    form_data = {tag["name"]: tag.get("value", "") for tag in soup.find_all("input", type="hidden") if tag.get("name")}

    m = re.search(
        r'interactiveReport\(\{.*?"ajaxIdentifier":"([^"]+)"',
        html,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not find APEX IR ajaxIdentifier in page source")
    # Unicode escapes in JSON-embedded JS strings
    ajax_id = m.group(1).replace(r"\u002F", "/")

    # Region static ID (e.g. "R47017706134393127895")
    rm = re.search(r"jQuery\('#(R\d+)_ir'\)", html) or re.search(r'"regionId":"(R\d+)"', html)
    if not rm:
        raise RuntimeError("Could not find APEX IR region ID in page source")
    rp = rm.group(1)

    return {
        "form_data": form_data,
        "ajax_id": ajax_id,
        "worksheet_id": soup.find("input", id=f"{rp}_worksheet_id")["value"],
        "report_id": soup.find("input", id=f"{rp}_report_id")["value"],
        "rows_per_page": int(soup.find("input", id=f"{rp}_row_select")["value"]),
    }


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------


def _parse_table(soup):
    """Return list of (code, name, holes, total) from the IR table.

    Expected column order: Code, Course, H1–H18, Par  (21 columns total).
    """
    rows = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 21:
                continue
            code = cells[0]
            name = cells[1]
            try:
                holes = [int(cells[i]) for i in range(2, 20)]
                total = int(cells[20])
            except (ValueError, IndexError):
                continue
            rows.append((code, name, holes, total))
    return rows


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def _next_pagination_mod(soup, current_min, rows_per_page):
    """Return the data-pagination value for the next page, or None if last page."""
    threshold = current_min + rows_per_page
    for btn in soup.find_all("button", attrs={"data-pagination": True}):
        mod = btn["data-pagination"]
        m = re.search(r"pgR_min_row=(\d+)", mod)
        if m and int(m.group(1)) >= threshold:
            return mod
    return None


def _ajax_next_page(session, state, pagination_mod):
    """POST an APEX IR pagination request; return (rows, next_pagination_mod)."""
    fd = state["form_data"]
    resp = session.post(
        AJAX_URL,
        data={
            "p_flow_id": fd["p_flow_id"],
            "p_flow_step_id": fd["p_flow_step_id"],
            "p_instance": fd["p_instance"],
            "p_request": "PLUGIN=" + state["ajax_id"],
            "p_widget_name": "worksheet",
            "p_widget_mod": "ACTION",
            "p_widget_action": "PAGE",
            "p_widget_action_mod": pagination_mod,
            "p_widget_num_return": state["rows_per_page"],
            "x01": state["worksheet_id"],
            "x02": state["report_id"],
            "pageItems": "",
        },
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "text/html, */*",
            "Referer": BASE_URL,
        },
        timeout=30,
    )
    resp.raise_for_status()
    page_soup = BeautifulSoup(resp.text, "html.parser")
    rows = _parse_table(page_soup)

    m = re.search(r"pgR_min_row=(\d+)", pagination_mod)
    current_min = int(m.group(1)) if m else 1
    next_mod = _next_pagination_mod(page_soup, current_min, state["rows_per_page"])
    return rows, next_mod


# ---------------------------------------------------------------------------
# Full fetch
# ---------------------------------------------------------------------------


def fetch_all_rows():
    """Fetch all (code, name, holes, total) across all pages."""
    session = requests.Session()
    session.headers.update(HEADERS)

    resp = session.get(BASE_URL, params={"tz": "America/New_York"}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    state = _extract_state(soup, resp.text)
    all_rows = _parse_table(soup)
    print(f"Page 1: {len(all_rows)} rows", file=sys.stderr)

    # First "next page" button is on the initial page
    btn = soup.find("button", attrs={"data-pagination": True})
    next_mod = btn["data-pagination"] if btn else None

    page = 2
    while next_mod:
        page_rows, next_mod = _ajax_next_page(session, state, next_mod)
        if not page_rows:
            break
        all_rows.extend(page_rows)
        print(f"Page {page}: {len(page_rows)} rows", file=sys.stderr)
        page += 1

    return all_rows


# ---------------------------------------------------------------------------
# Course name normalisation
# ---------------------------------------------------------------------------


def _build_course_name(code, name, easy_names):
    """Derive a consistent course name.

    Easy courses keep the site name as-is.
    Hard courses use the paired easy-course name + ' - Hard', so that
    abbreviated hard names (e.g. "20,000 Leagues Hard") become the full
    easy name + ' - Hard' (e.g. "20,000 Leagues Under The Sea - Hard").
    """
    if code.endswith("H"):
        prefix = code[:-1]
        if prefix in easy_names:
            return easy_names[prefix] + " - Hard"
        # Fallback: normalise "Foo Hard" → "Foo - Hard"
        return re.sub(r"(?<! -)\s*\bHard\s*$", " - Hard", name).strip()
    return name


def build_csv_rows(raw_rows):
    """Convert raw rows into sorted CSV rows with normalised course names."""
    easy_names = {code[:-1]: name for code, name, _holes, _total in raw_rows if code.endswith("E")}

    csv_rows = []
    for code, name, holes, total in raw_rows:
        course_name = _build_course_name(code, name, easy_names)
        csv_rows.append([course_name, code] + holes + [total])

    csv_rows.sort(key=lambda r: r[0].casefold())
    return csv_rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["course", "code"] + HOLE_NUMBERS + ["total"])
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} courses to {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fetch par data from mywmgt.com and write pars.csv")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    raw = fetch_all_rows()
    print(f"Total raw rows fetched: {len(raw)}", file=sys.stderr)

    if not raw:
        print("No data fetched.", file=sys.stderr)
        sys.exit(1)

    csv_rows = build_csv_rows(raw)
    write_csv(csv_rows, args.output)


if __name__ == "__main__":
    main()
