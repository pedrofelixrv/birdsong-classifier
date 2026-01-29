import os
import re
import time
import requests
import pandas as pd
from pathlib import Path

API_BASE = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.getenv("XC_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina a variável de ambiente XC_API_KEY.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

# ===================== UTIL =====================

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\s\-\.]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s

def normalize_species_for_api(s: str) -> str:
    """Converte 'turdus_rufiventris' -> 'turdus rufiventris' e remove espaços extras."""
    s = str(s).strip()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s

def parse_length_to_seconds(length_str):
    try:
        m, s = str(length_str).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None

def extract_error_info(err):
    if err is None:
        return None, None
    if isinstance(err, dict):
        e = err.get("error", err)
        if isinstance(e, dict):
            return e.get("code"), e.get("message")
        return None, str(e)
    return None, str(err)

# ===================== API =====================

def xc_search(query: str, page: int = 1, per_page: int = 100):
    r = requests.get(
        API_BASE,
        params={"query": query, "key": API_KEY, "page": page, "per_page": per_page},
        headers=HEADERS,
        timeout=30,
    )

    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": {"code": "non_json_error", "message": r.text[:500]}}
        return None, r.status_code, err

    return r.json(), 200, None

def download_file(url: str, out_path: Path):
    if url.startswith("//"):
        url = "https:" + url

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60, headers=HEADERS) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

# ===================== CORE =====================

def fetch_and_download_species(
    species_full: str,
    out_dir: Path,
    log_rows: list,
    max_files: int = 20,
    quality_min: str = "B",
    only_song: bool = True,
    per_page: int = 100,
    sleep_s: float = 0.3,
):
    species_raw = str(species_full).strip()
    if not species_raw:
        log_rows.append({
            "scientific_name": "",
            "scientific_name_raw": species_raw,
            "query_used": None,
            "xc_id": None,
            "status": "invalid_empty",
            "http_status": None,
            "error_code": None,
            "error_message": "Nome vazio",
        })
        return 0

    species_api = normalize_species_for_api(species_raw)

    if len(species_api.split()) < 2:
        log_rows.append({
            "scientific_name": species_api,
            "scientific_name_raw": species_raw,
            "query_used": None,
            "xc_id": None,
            "status": "invalid_name_format",
            "http_status": None,
            "error_code": None,
            "error_message": "Nome científico precisa ter pelo menos 2 palavras (Genus species).",
        })
        return 0

    # ✅ use ESPAÇO entre tags (em vez de '+')
    query_parts = [f'sp:"{species_api}"']
    if only_song:
        query_parts.append("type:song")
    query = " ".join(query_parts)

    downloaded = 0
    page = 1
    logged_no_results = False

    while downloaded < max_files:
        data, http_status, err = xc_search(query=query, page=page, per_page=per_page)

        if data is None:
            code, msg = extract_error_info(err)
            log_rows.append({
                "scientific_name": species_api,
                "scientific_name_raw": species_raw,
                "query_used": query,
                "xc_id": None,
                "quality": None,
                "type": None,
                "country": None,
                "locality": None,
                "date": None,
                "length_sec": None,
                "samplerate": None,
                "license": None,
                "file_path": None,
                "status": "search_error",
                "http_status": http_status,
                "error_code": code,
                "error_message": msg,
            })
            return downloaded

        recs = data.get("recordings", [])
        if not recs:
            if not logged_no_results:
                log_rows.append({
                    "scientific_name": species_api,
                    "scientific_name_raw": species_raw,
                    "query_used": query,
                    "xc_id": None,
                    "status": "no_results",
                    "http_status": 200,
                    "error_code": None,
                    "error_message": None,
                })
                logged_no_results = True
            break

        for rec in recs:
            if downloaded >= max_files:
                break

            q = (rec.get("q") or "").strip().upper()
            if quality_min and q and q > quality_min:
                continue

            rec_id = rec.get("id")
            file_url = rec.get("file")
            if not rec_id or not file_url:
                continue

            folder = out_dir / safe_name(species_api)
            out_path = folder / f"XC{rec_id}.mp3"

            status = "downloaded"
            err_msg = None

            try:
                if not out_path.exists():
                    download_file(file_url, out_path)
                    time.sleep(sleep_s)
                else:
                    status = "skipped_exists"
            except Exception as e:
                status = "download_error"
                err_msg = str(e)

            log_rows.append({
                "scientific_name": species_api,
                "scientific_name_raw": species_raw,
                "query_used": query,
                "xc_id": rec_id,
                "quality": q,
                "type": rec.get("type"),
                "country": rec.get("cnt"),
                "locality": rec.get("loc"),
                "date": rec.get("date"),
                "length_sec": parse_length_to_seconds(rec.get("length", "")),
                "samplerate": rec.get("smp"),
                "license": rec.get("lic"),
                "file_path": str(out_path),
                "status": status,
                "http_status": 200,
                "error_code": None,
                "error_message": err_msg,
            })

            if status == "downloaded":
                downloaded += 1

        page += 1
        if page > int(data.get("numPages", page)):
            break

    return downloaded

# ===================== MAIN =====================

def main():
    csv_path = "especies_itapaje.csv"
    species_col = "nome_especie"
    out_dir = Path("downloads")

    max_files_per_species = 100
    quality_min = "C"
    only_song = True
    per_page = 100

    df = pd.read_csv(csv_path)

    species_list = (
        df[species_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    species_list = species_list[species_list != ""].drop_duplicates()

    log_rows = []

    for sp in species_list:
        n = fetch_and_download_species(
            species_full=sp,
            out_dir=out_dir,
            log_rows=log_rows,
            max_files=max_files_per_species,
            quality_min=quality_min,
            only_song=only_song,
            per_page=per_page,
        )
        print(f"{sp} (API: {normalize_species_for_api(sp)}) -> {n} baixados (meta {max_files_per_species})")

        # salva log incremental
        pd.DataFrame(log_rows).to_csv("download_log.csv", index=False)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv("download_log.csv", index=False)

    summary = (
        log_df.groupby(["scientific_name", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["scientific_name", "count"], ascending=[True, False])
    )
    summary.to_csv("summary_by_species.csv", index=False)

    print("\nArquivos gerados:")
    print(" - download_log.csv (detalhado)")
    print(" - summary_by_species.csv (resumo por espécie/status)")
    print(f"Downloads em: {out_dir.resolve()}")

if __name__ == "__main__":
    main()