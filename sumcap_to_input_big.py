import os
import re
import pandas as pd


def parse_relistat(filepath):

    M_cases = {}
    TH_events = {}

    with open(filepath, "r", encoding="utf-8") as f:

        lines = [line.strip() for line in f if line.strip()]

    m_buffer = []
    current_M = None

    for line in lines:
        
        if line.startswith("# M"):
            
            m_ids = re.findall(r"M\w+", line)
            if not m_ids:
                continue

            for mid in m_ids:
                current_M = mid
                if mid not in M_cases:
                    M_cases[mid] = []
                m_buffer.append(mid)

        elif line.startswith("# E") and current_M:
            
            g_list = re.findall(r"G\d+", line)
            if g_list:
                M_cases[current_M].extend(g_list)

        
        elif line.startswith("# TH"):
            
            th_ids = re.findall(r"TH\d+", line)
            if th_ids:
                th = th_ids[0]
                TH_events[th] = m_buffer.copy()
                m_buffer = []
                current_M = None


    return M_cases, TH_events


def process_sumcap(filepath):

    df = pd.read_csv(filepath, sep="\t", engine="python", dtype=str)


    for col in ["nev", "dir", "class", "sumcap"]:
        if col not in df.columns:
            raise KeyError(f"Hiányzó oszlop a sumcap fájlban: {col} ({filepath})")


    df = df[df["class"].str.lower() == "x"].copy()

    df["sumcap"] = pd.to_numeric(df["sumcap"], errors="coerce").fillna(0)

    def make_key(row):

        nev = str(row["nev"])
        base = nev.split()[0]
        if "-" not in base:
            return None

        a, b = base.split("-")

        d = str(row["dir"]).upper()
        if d == "FW":
            return f"{a}_{b}"
        elif d == "BW":
            return f"{b}_{a}"
        else:
            return None

    df["link_key"] = df.apply(make_key, axis=1)
    df = df.dropna(subset=["link_key"])


    agg = df.groupby("link_key")["sumcap"].sum()

    return agg.to_dict()



def build_dataset(relistat_path, sumcap_folder):

    M_cases, TH_events = parse_relistat(relistat_path)

    optical_cols = sorted({g for g_list in M_cases.values() for g in g_list})

    th_to_file = {}
    for f in os.listdir(sumcap_folder):
        low = f.lower()
        if not low.startswith("sumcap"):
            continue

        full_path = os.path.join(sumcap_folder, f)

        m = re.search(r"th\d+", low)
        if m:
            th_id = m.group(0).upper()
        else:
            th_id = "TH0"

        th_to_file[th_id] = full_path


    th_features = {}
    all_links = set()

    for th, m_list in TH_events.items():
        if th not in th_to_file:
            print(f"Figyelmeztetés: nincs sumcap fájl ehhez a TH-hoz: {th}")
            continue

        feats = process_sumcap(th_to_file[th])
        th_features[th] = feats
        all_links.update(feats.keys())

    all_links = sorted(all_links)

    rows = []

    for th, m_list in TH_events.items():
        feats = th_features.get(th, {})

        for m in m_list:
            row = {
                "TH": th,
                "M_case": m,
            }

            g_list = M_cases.get(m, [])
            for g in optical_cols:
                row[g] = 1 if g in g_list else 0

            for lnk in all_links:
                row[lnk] = feats.get(lnk, 0)

            rows.append(row)

    return pd.DataFrame(rows)



if __name__ == "__main__":

    relistat_path = "E:\.szakdoga\Reli_Stat_Big\ReliStat.txt"
    sumcap_folder = "E:\.szakdoga\Results_big"

    df = build_dataset(relistat_path, sumcap_folder)
    df.to_csv("ml_ready_dataset_big.csv", index=False, encoding="utf-8")
    print("Kész: ml_ready_dataset_big.csv")
