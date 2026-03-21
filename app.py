import os
import re
import time
import subprocess
from datetime import datetime, timedelta
from html.parser import HTMLParser
from collections import defaultdict

import psutil
import streamlit as st


DATA_DIR = "data"
REPORTS_DIR = os.path.join(DATA_DIR, "windows_reports")
BATTERY_REPORT = os.path.join(REPORTS_DIR, "battery-report.html")

DEFAULT_DAYS = 7


# ---------- Helpers ----------
def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_dt_any(s: str):
    s = (s or "").strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?)", s)
    if m:
        for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(m.group(1), f)
            except Exception:
                pass
    return None

def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---------- Windows report ----------
def generate_battery_report():
    ensure_dirs()
    subprocess.run(["powercfg", "/batteryreport", "/output", BATTERY_REPORT], check=True)

def parse_capacity_wear(html: str):
    def extract_mwh(label):
        idx = html.upper().find(label.upper())
        if idx == -1:
            return None
        chunk = html[idx: idx + 4000]
        m = re.search(r"(\d[\d,]*)\s*mWh", chunk, re.IGNORECASE)
        if not m:
            return None
        return int(m.group(1).replace(",", ""))

    design = extract_mwh("DESIGN CAPACITY")
    full = extract_mwh("FULL CHARGE CAPACITY")
    if design and full:
        wear = (1 - full / design) * 100
        return {"design_mwh": design, "full_mwh": full, "wear_pct": max(0.0, wear)}
    return None


class TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = self.in_tr = self.in_cell = False
        self.tables = []
        self.cur_table, self.cur_row, self.cur_cell = [], [], []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self.in_table = True
            self.cur_table = []
        elif tag == "tr" and self.in_table:
            self.in_tr = True
            self.cur_row = []
        elif tag in ("td", "th") and self.in_tr:
            self.in_cell = True
            self.cur_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            text = re.sub(r"\s+", " ", "".join(self.cur_cell)).strip()
            self.cur_row.append(text)
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if any(c.strip() for c in self.cur_row):
                self.cur_table.append(self.cur_row)
        elif tag == "table" and self.in_table:
            self.in_table = False
            if self.cur_table:
                self.tables.append(self.cur_table)

    def handle_data(self, data):
        if self.in_cell:
            self.cur_cell.append(data)


def extract_usage_events_from_any_table(tables):
    events = []
    for t in tables:
        if not t or len(t) < 2:
            continue
        for row in t[1:]:
            if not row:
                continue

            dt = None
            for cell in row:
                dt = parse_dt_any(cell)
                if dt:
                    break
            if not dt:
                continue

            cap = None
            for cell in row:
                m = re.search(r"(\d[\d,]*)\s*mWh", cell, re.IGNORECASE)
                if m:
                    cap = int(m.group(1).replace(",", ""))
                    break
            if cap is None:
                continue

            source = None
            for cell in row:
                c = cell.lower().strip()
                if "battery" in c:
                    source = "Battery"
                    break
                if c in ("ac", "a/c"):
                    source = "AC"
                    break
                if "ac" in c and "capacity" not in c:
                    source = "AC"
            if source is None:
                continue

            events.append({"dt": dt, "source": source, "cap_mwh": cap})

    events.sort(key=lambda e: e["dt"])
    # de-dup
    seen = set()
    dedup = []
    for e in events:
        k = (e["dt"], e["source"], e["cap_mwh"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(e)
    return dedup


def summarize_last_days(events, days=7):
    if not events:
        return None
    cutoff = datetime.now() - timedelta(days=days)
    ev = [e for e in events if e["dt"] >= cutoff]
    if len(ev) < 2:
        ev = events[-50:] if len(events) >= 2 else events
    if len(ev) < 2:
        return None

    max_cap = max(e["cap_mwh"] for e in ev)
    high_charge_threshold = 0.90 * max_cap

    battery_minutes = 0.0
    ac_minutes = 0.0
    high_charge_ac_minutes = 0.0
    drain_rates = []
    seg_counts = defaultdict(int)

    for i in range(1, len(ev)):
        prev, cur = ev[i - 1], ev[i]
        dt_s = (cur["dt"] - prev["dt"]).total_seconds()
        if dt_s <= 0 or dt_s > 6 * 3600:
            continue

        on_batt = (cur["source"].lower() == "battery")
        if on_batt:
            battery_minutes += dt_s / 60
            seg_counts["battery"] += 1
        else:
            ac_minutes += dt_s / 60
            seg_counts["ac"] += 1
            if cur["cap_mwh"] >= high_charge_threshold:
                high_charge_ac_minutes += dt_s / 60

        if prev["source"].lower() == "battery" and cur["source"].lower() == "battery":
            if cur["cap_mwh"] <= prev["cap_mwh"]:
                drop = prev["cap_mwh"] - cur["cap_mwh"]
                per_hr = drop * (3600 / dt_s)
                drain_rates.append(per_hr)

    return {
        "events_used": len(ev),
        "battery_minutes": battery_minutes,
        "ac_minutes": ac_minutes,
        "battery_segments": seg_counts.get("battery", 0),
        "ac_segments": seg_counts.get("ac", 0),
        "high_charge_ac_minutes": high_charge_ac_minutes,
        "drain_avg_mwh_hr": mean(drain_rates),
        "cutoff": cutoff,
    }


# ---------- Live snapshot ----------
def live_snapshot():
    b = psutil.sensors_battery()
    cpu = psutil.cpu_percent(interval=0.2)
    ram = psutil.virtual_memory().percent
    out = {
        "cpu": cpu,
        "ram": ram,
        "battery_percent": None,
        "plugged": None
    }
    if b:
        out["battery_percent"] = float(b.percent)
        out["plugged"] = bool(b.power_plugged)
    return out


def quick_recs(live, wear, seven):
    recs = []
    if live["battery_percent"] is not None:
        if live["plugged"] and live["battery_percent"] >= 80:
            recs.append("Plugged in above ~80% → unplugging earlier can reduce long-term wear.")
        if (not live["plugged"]) and live["battery_percent"] <= 20:
            recs.append("On battery below ~20% → plug in soon to avoid deep discharge.")
    if live["cpu"] >= 70:
        recs.append("CPU is high right now → check Task Manager for a runaway app.")
    if wear and wear["wear_pct"] >= 20:
        recs.append(f"Battery wear ≈ {wear['wear_pct']:.0f}% → if battery life is bad, replacement is a real option.")
    if seven and seven["high_charge_ac_minutes"] >= 300:
        recs.append("You spend a lot of time on AC at high charge → consider unplugging earlier when possible.")
    if not recs:
        recs.append("No obvious red flags right now.")
    return recs


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Laptop Analytics (Windows 11)", layout="wide")
ensure_dirs()

st.title("Laptop Analytics Dashboard (Windows 11)")
st.caption("Open-and-check dashboard: live stats + Windows stored battery history (no need to keep it running).")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Live (right now)")
    live = live_snapshot()
    st.write(f"**Time:** {now_str()}")
    st.metric("CPU (%)", f"{live['cpu']:.1f}")
    st.metric("RAM (%)", f"{live['ram']:.1f}")

    if live["battery_percent"] is None:
        st.info("Battery info not available on this device.")
    else:
        st.metric("Battery (%)", f"{live['battery_percent']:.0f}")
        st.write("**Power:** " + ("PLUGGED" if live["plugged"] else "ON BATTERY"))

with colB:
    st.subheader("Windows history (batteryreport)")
    if st.button("Refresh batteryreport (powercfg)"):
        try:
            generate_battery_report()
            st.success("batteryreport updated.")
        except Exception as e:
            st.error(f"batteryreport failed: {e}")

    if os.path.exists(BATTERY_REPORT):
        html = read_file(BATTERY_REPORT)
        wear = parse_capacity_wear(html)
        if wear:
            st.write("**Battery Health (from report)**")
            st.write(f"- Design capacity: {wear['design_mwh']:,} mWh")
            st.write(f"- Full charge cap: {wear['full_mwh']:,} mWh")
            st.write(f"- Estimated wear: **{wear['wear_pct']:.1f}%**")
        else:
            wear = None
            st.warning("Could not parse capacity/wear yet.")

        parser = TableHTMLParser()
        parser.feed(html)
        events = extract_usage_events_from_any_table(parser.tables)
        seven = summarize_last_days(events, days=DEFAULT_DAYS)

        st.write("")
        st.write(f"Tables found: **{len(parser.tables)}** | Parsed usage events: **{len(events)}**")

        if seven:
            st.write(f"**Last {DEFAULT_DAYS} days (whatever Windows has):**")
            st.write(f"- On battery: {seven['battery_minutes']:.0f} min")
            st.write(f"- On AC: {seven['ac_minutes']:.0f} min")
            st.write(f"- High-charge AC time (approx): {seven['high_charge_ac_minutes']:.0f} min")
            st.write(f"- Avg drain rate (battery intervals): {seven['drain_avg_mwh_hr']:.0f} mWh/hr")
        else:
            st.warning("Couldn’t summarize history yet (may be missing usage rows in this report layout).")
    else:
        wear = None
        seven = None
        st.info("No batteryreport yet. Click refresh to generate it.")

st.divider()
st.subheader("Recommendations")
for r in quick_recs(live, wear, seven):
    st.write(f"- {r}")

st.divider()
with st.expander("Open report file"):
    if os.path.exists(BATTERY_REPORT):
        st.write("Report path:")
        st.code(os.path.abspath(BATTERY_REPORT))
        st.write("Tip: open it in your browser to confirm it contains the 'Battery usage' section.")
    else:
        st.write("Generate the report first.")

# Auto refresh option (lightweight)
auto = st.checkbox("Auto-refresh page every 5 seconds", value=False)
if auto:
    time.sleep(5)
    st.rerun()
