import os
import re
import pandas as pd
from collections import defaultdict

def parse_relistat(filepath):
    M_cases = {}
    TH_events = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    m_buffer = []
    current_M = None

    for line in lines:
        if line.startswith("# M"):
            m_id = re.findall(r"M\d+", line)[0]
            current_M = m_id
            M_cases[m_id] = []
            m_buffer.append(m_id)

        elif line.startswith("# E") and current_M:
            g = re.findall(r"G\d+", line)
            if g:
                M_cases[current_M].append(g[0])

        elif line.startswith("# TH"):
            th = re.findall(r"TH\d+", line)[0]
            TH_events[th] = m_buffer.copy()
            m_buffer = []
            current_M = None

    return M_cases, TH_events


def process_sumcap(filepath):
    df = pd.read_csv(filepath, sep="\t", engine="python", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["nev", "dir", "class", "sumcap"]:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in {filepath}")

    df = df[df["class"].str.lower() == "x"]
    df["sumcap"] = pd.to_numeric(df["sumcap"], errors="coerce").fillna(0)

    def link_dir(row):
        val = row["nev"].split()[0]
        if "-" not in val:
            return None, None
        a, b = val.split("-")
        if row["dir"].upper() == "FW":
            return a[0] + b[0], row["sumcap"]
        elif row["dir"].upper() == "BW":
            return b[0] + a[0], row["sumcap"]
        return None, None

    data = {}
    for _, r in df.iterrows():
        k, v = link_dir(r)
        if k:
            data[k] = data.get(k, 0) + v
    return data


def build_dataset(relistat_path, sumcap_folder):
    M_cases, TH_events = parse_relistat(relistat_path)
    optical_cols = [f"G{i}" for i in range(1, 8)]
    all_links = set()

    th_to_file = {}
    for f in os.listdir(sumcap_folder):
        if f.lower().startswith("sumcap_th"):
            th = re.findall(r"th\d+", f.lower())[0].upper()
            th_to_file[th] = os.path.join(sumcap_folder, f)

    th_features = {}
    for th, m_list in TH_events.items():
        if th in th_to_file:
            feats = process_sumcap(th_to_file[th])
            th_features[th] = feats
            all_links.update(feats.keys())
        else:
            print(f"Missing sumcap file for {th}")

    all_links = sorted(all_links)


    rows = []
    for th, m_list in TH_events.items():
        for m in m_list:
            g_list = M_cases.get(m, [])
            row = {"TH": th, "M_case": m}

            for g in optical_cols:
                row[g] = 1 if g in g_list else 0

            feats = th_features.get(th, {})
            for lnk in all_links:
                row[lnk] = feats.get(lnk, 0)

            rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    relistat_path = "E:/.szakdoga/Reli_Stat/ReliStat.txt"
    sumcap_folder = "E:/.szakdoga/Results"

    df = build_dataset(relistat_path, sumcap_folder)
    df.to_csv("ml_ready_dataset.csv", index=False)
    print("Saved to ml_ready_dataset.csv")
