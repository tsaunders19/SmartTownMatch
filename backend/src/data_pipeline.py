from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from shapely.geometry import Point
from dotenv import load_dotenv
import time
from typing import Tuple, Dict
from io import StringIO
import logging
import zipfile
import fnmatch
import re

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEBUG_LOG_PATH = Path("data/raw/pipeline_debug.log")
DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_file_handler = logging.FileHandler(DEBUG_LOG_PATH, mode="w")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(_file_handler)

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CACHE_PATH = RAW_DIR / "api_cache.json"
FILE_SOURCES_PATH = RAW_DIR / "file_sources.json"

# TODO self: have to fix the backend freezing for minutes when the API cache is too big
try:
    with open(CACHE_PATH, "r") as fp:
        _API_CACHE = json.load(fp)
        logger.info(f"[cache] Loaded API cache with {len(_API_CACHE)} entries from {CACHE_PATH}")
except Exception:
    _API_CACHE = {}
    logger.info("[cache] No existing API cache. Starting fresh.")

try:
    with open(FILE_SOURCES_PATH, "r") as fp:
        _FILE_SOURCES = json.load(fp)
except Exception:
    _FILE_SOURCES = {}

def _normalize_town_name(name) -> str:
    """Standardize town names for merging. Accepts str or NaN."""
    if not isinstance(name, str):
        return name

    return (
        name.upper()
        .strip()
        .replace(" CITY", "")
        .replace(" TOWN", "")
        .replace("CENSUS DESIGNATED PLACE", "")
    )

def _save_cache():
    try:
        _ensure_dir(CACHE_PATH)
        with open(CACHE_PATH, "w") as fp:
            json.dump(_API_CACHE, fp)
    except Exception as exc:
        logger.warning(f"[cache] Failed to save cache: {exc}")

def _save_file_sources():
    try:
        _ensure_dir(FILE_SOURCES_PATH)
        with open(FILE_SOURCES_PATH, "w") as fp:
            json.dump(_FILE_SOURCES, fp, indent=2)
    except Exception as exc:
        logger.warning(f"[file_sources] Failed to save metadata: {exc}")

MASSGIS_TOWNS_URL = (
    "https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/census2020/CENSUS2020TOWNS_SHP.zip"
)
MASSGIS_TOWNS_LOCAL = RAW_DIR / "CENSUS2020TOWNS_SHP.zip"
MASSGIS_TOWNS_ZIP_URL = MASSGIS_TOWNS_URL

ACS_API_URL = "https://api.census.gov/data/2022/acs/acs5"  # latest 5-year
ACS_VARIABLES = {
    "NAME": "TownName",
    "B01003_001E": "Population",  # Total population
    "B19013_001E": "MedianIncome",  # Median household income
    "B25077_001E": "MedianHomePrice",  # Median home price
}

# NOTE: Using county subdivision (state-county-subdivision GEOID) for MA
# 050 = county, 060 = county subdivision (town); but API table differs.
ACS_FOR = "county subdivision:*"  # used with &in=state:25

FALLBACK_TOWNS_URL = None  # TODO self: remove this during clean up

MASSGIS_DISTRICTS_URL = (
    "https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/schooldistricts.zip"
)


# NOTE: point to the public Profiles HTML view and scrape the primary table with pandas.
# A local HTML cache is kept to avoid repeated downloads.
# the scraping logic is isolated inside fetch_education_data()
# TODO ASAP: there is an export button on the page that downloads a csv but i cant get the link for some reason. work on parsing that instead

DESE_PPX_URL = "https://profiles.doe.mass.edu/statereport/ppx.aspx?fy=2023"  # Per-pupil expenditure (FY2023)
# MCAS district-level exports are now behind interactive dashboards that i dont know how to scrape
# For now EducationScore is based solely on normalised PPX.
DESE_MCAS_URL = None

# FBI Uniform Crime Report (UCR) – 2019 crimes by city/town (Table 6)
FBI_UCR_URL = "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-6/table-6/output.xls"
# Massachusetts – Offenses Known to Law Enforcement by City (Table 8)
FBI_MA_CITY_URL = "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-8/table-8-state-cuts/massachusetts.xls/output.xls"

DATA_SOURCE_URLS = {
    "MASSGIS_TOWNS": MASSGIS_TOWNS_URL,
    "ACS_API": ACS_API_URL,
    "MASSGIS_SCHOOL_DISTRICTS": MASSGIS_DISTRICTS_URL,
    "DESE_PPX": DESE_PPX_URL,
    # NOTE: MCAS URL is None for now
    **({} if DESE_MCAS_URL is None else {"DESE_MCAS": DESE_MCAS_URL}),
    "FBI_UCR": FBI_UCR_URL,
    "FBI_MA_CITY": FBI_MA_CITY_URL,
    # NOTE: API endpoints Walk Score & Google Places are not included here because they are queried dynamically per-town need to inspect helper functions instead.
}

print("[data_pipeline] External data sources:")
for name, link in DATA_SOURCE_URLS.items():
    print(f"  • {name}: {link}")

WALK_SCORE_API_KEY = os.getenv("WALK_SCORE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

AMENITY_TYPES = [
    "park",
    "restaurant",
    "cafe",
    "movie_theater",
    "library",
    "bar",
]


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download *url* to *dest* if it does not already exist. Try fallback if fails.

    A metadata registry (`file_sources.json`) tracks which original URL was
    used for every file so that subsequent runs can clearly report the exact
    source, even when the local cached copy is reused.
    """
    if dest.exists():
        origin = _FILE_SOURCES.get(str(dest), "UNKNOWN")
        print(f"[download_file] Using cached file {dest} (source: {origin})")
        return dest

    _ensure_dir(dest)
    urls = [url]

    for attempt_url in urls:
        try:
            print(f"[download_file] Downloading {attempt_url} -> {dest} …")
            r = requests.get(attempt_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as fp:
                for chunk in r.iter_content(chunk_size):
                    fp.write(chunk)
            _FILE_SOURCES[str(dest)] = attempt_url
            _save_file_sources()
            print(f"[download_file] SUCCESS: fetched {attempt_url}")
            return dest
        except Exception as exc:
            print(f"[download_file] WARN: download failed ({exc}). Trying next fallback…")
    raise RuntimeError(f"Failed to download {url} or fallbacks")


def load_towns_geo() -> gpd.GeoDataFrame:
    """Load MA town boundaries exclusively from the official 2020 shapefile ZIP."""
    zip_path = download_file(
        MASSGIS_TOWNS_ZIP_URL, RAW_DIR / "CENSUS2020TOWNS_SHP.zip"
    )

    # NOTE: geopandas/fiona can read directly from zip using the scheme: zip://…!<shp>
    with zipfile.ZipFile(zip_path, "r") as zf:
        shapefile_name = next(
            (n for n in zf.namelist() if fnmatch.fnmatch(n.lower(), "*poly.shp")), None
        )
        if shapefile_name is None:
                    raise RuntimeError("Shapefile (.shp) not found in ZIP")

        gdf = gpd.read_file(f"zip://{zip_path}!{shapefile_name}")

    town_col = next(
        (
            c
            for c in gdf.columns
            if c.lower() in {"town", "town_name", "name", "town20"}
        ),
        None,
    )
    if town_col is None:
        raise RuntimeError("Could not identify town name column in shapefile.")
    gdf = gdf.rename(columns={town_col: "TownName"})

    geoid_col = next((c for c in gdf.columns if "GEOID" in c.upper()), None)
    if geoid_col:
        gdf = gdf.rename(columns={geoid_col: "GEOID"})

    county_col = next((c for c in gdf.columns if "county" in c.lower()), None)
    if county_col:
        gdf = gdf.rename(columns={county_col: "County"})

    if "SQ_MILES" not in gdf.columns:
        gdf["SQ_MILES"] = gdf.geometry.to_crs(epsg=3395).area / 2_589_988.11

    gdf["TownName"] = gdf["TownName"].apply(_normalize_town_name)
    
    print(f"Loaded {len(gdf)} town geometries.")
    return gdf[[col for col in ["TownName", "County", "TOWN_ID", "SQ_MILES", "geometry"] if col in gdf.columns]]


def fetch_acs_data(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch ACS variables via Census API.
    """
    key = api_key or os.getenv("CENSUS_API_KEY")
    params = {
        "get": ",".join(ACS_VARIABLES.keys()),
        "for": ACS_FOR,
        "in": "state:25",  # Massachusetts
    }
    if key:
        params["key"] = key
    print("[fetch_acs_data] Querying ACS API …")
    r = requests.get(ACS_API_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns=ACS_VARIABLES)

    for col in ["Population", "MedianIncome", "MedianHomePrice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # NOTE: The ACS NAME value looks like "Boston city, Suffolk County, Massachusetts".
    # Take just the segment before the first comma
    df["TownName"] = df["TownName"].str.split(",").str[0]

    df["TownName"] = df["TownName"].apply(_normalize_town_name)

    print(f"Loaded {len(df)} town records from ACS.")
    return df

def _ensure_lat_lon(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure df has 'lat' and 'lon' columns using centroid if missing."""
    if "lat" not in df.columns or "lon" not in df.columns:
        centroids = df.geometry.to_crs(epsg=4326).centroid
        df["lat"] = centroids.y
        df["lon"] = centroids.x
    return df

def _walk_score_request(lat: float, lon: float, address: str) -> Dict[str, Optional[int]]:
    """Request walk, transit, and bike scores from the official Walk Score API."""
    cache_key = f"walkscore2:{lat:.5f}:{lon:.5f}"
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    url = "https://api.walkscore.com/score"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "address": address,
        "transit": 1,
        "bike": 1,
        "wsapikey": WALK_SCORE_API_KEY,
    }
    for attempt in range(3):
        try:
            if attempt > 0:
                print(f"[walkscore] Retry {attempt} for ({lat:.4f}, {lon:.4f}) …")
            print(f"[walkscore] Remote request for ({lat:.4f}, {lon:.4f}) …")
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
            result = {
                "walk": data.get("walkscore"),
                "transit": (data.get("transit") or {}).get("score"),
                "bike": (data.get("bike") or {}).get("score"),
            }
            _API_CACHE[cache_key] = result
            _save_cache()
            return result
        except Exception as exc:
            print(f"[walkscore] WARN: {exc}")
            time.sleep(1 + attempt)

    _API_CACHE[cache_key] = {"walk": None, "transit": None, "bike": None}
    _save_cache()
    return {"walk": None, "transit": None, "bike": None}


def add_walk_scores(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not WALK_SCORE_API_KEY:
        raise RuntimeError("WALK_SCORE_API_KEY missing. Add it to .env file.")
    df = _ensure_lat_lon(df)
    walk_scores = []
    transit_scores = []
    bike_scores = []
    for _, row in df.iterrows():
        res = _walk_score_request(row["lat"], row["lon"], row["TownName"])
        walk_scores.append(res.get("walk"))
        transit_scores.append(res.get("transit"))
        bike_scores.append(res.get("bike"))
        time.sleep(0.2)
    df["WalkScore"] = walk_scores
    df["TransitScore"] = transit_scores
    df["BikeScore"] = bike_scores
    return df

def _places_count(lat: float, lon: float, place_type: str, radius_m: int = 5000) -> int:
    cache_key = f"places:{place_type}:{lat:.5f}:{lon:.5f}:{radius_m}"
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_PLACES_API_KEY,
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "type": place_type,
    }
    try:
        print(f"[places_count] Remote request for {place_type} at ({lat:.4f}, {lon:.4f}) …")
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        count = len(data.get("results", []))
        _API_CACHE[cache_key] = count
        _save_cache()
        return count
    except Exception as exc:
        print(f"[places_count] WARN: {exc}")
        return 0


def add_amenities_score(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate a composite 'AmenitiesScore' from Google Places API. Uses caching to minimise calls."""
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError("GOOGLE_PLACES_API_KEY missing. Add it to .env file.")
    df = _ensure_lat_lon(df)

    amenity_counts = {row_index: 0 for row_index in df.index}

    for amenity in AMENITY_TYPES:
        print(f"[amenities] Processing amenity type '{amenity}' …")
        for _, row in df.iterrows():
            count = _places_count(row["lat"], row["lon"], amenity)
            amenity_counts[row.name] += count
            time.sleep(0.05)
    df["AmenitiesScore"] = pd.Series(amenity_counts)
    print("[amenities] Done calculating AmenitiesScore.")
    return df


def fetch_education_data() -> pd.DataFrame:
    """Generate *EducationScore* for every Massachusetts town.

    The DESE finance portal lists **per-pupil expenditures (PPX)** by district in
    an online HTML table. We scrape that table (or reuse a cached copy in
    `data/raw/ppx2023.html`), normalise the PPX values (higher spending ⇒ higher
    score), and then map each district back to its member towns using the
    MassGIS *Public School Districts* layer.

    Returns
    -------
    pd.DataFrame
        Columns: `TownName`, `EducationScore` (0-1 scaled)
    """

    print("[fetch_education_data] Fetching DESE per-pupil expenditure data …")

    cache_path = RAW_DIR / "ppx2023.html"
    if cache_path.exists():
        print(f"[fetch_education_data] Using cached HTML {cache_path}")
        html = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        r = requests.get(DESE_PPX_URL, timeout=60)
        r.raise_for_status()
        html = r.text
        _ensure_dir(cache_path)
        cache_path.write_text(html, encoding="utf-8")
        _FILE_SOURCES[str(cache_path)] = DESE_PPX_URL
        _save_file_sources()

    try:
        df_list = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        raise RuntimeError("No tables found in DESE PPX page. Structure may have changed.")

    if df_list:
        df = df_list[0]
        df.columns = [
            "District Name",
            "ORG4CODE",
            "In-District Expenditures",
            "Total In-district FTEs",
            "In-District Expenditures per Pupil",
            "Total Expenditures",
            "Total Pupil FTEs",
            "PerPupilExpenditure",
        ]
        df["TownName"] = df["District Name"].apply(_normalize_town_name)
        df_edu = df[["TownName", "ORG4CODE", "PerPupilExpenditure"]].copy()

        df_edu["PerPupilExpenditure"] = (
            df_edu["PerPupilExpenditure"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df_edu["PerPupilExpenditure"] = pd.to_numeric(df_edu["PerPupilExpenditure"], errors="coerce")
        print(f"[debug] Sample PPX numeric values: {df_edu['PerPupilExpenditure'].dropna().head().tolist()}")
    else:
        print("[fetch_education_data] WARN: Could not find table in HTML.")

    district_zip_path = RAW_DIR / "schooldistricts.zip"
    download_file(MASSGIS_DISTRICTS_URL, district_zip_path)

    try:
        with zipfile.ZipFile(district_zip_path, 'r') as zf:
            shp_name = next(
                (
                    n
                    for n in zf.namelist()
                    if fnmatch.fnmatch(n.upper(), "*SCHOOLDISTRICTS_POLY.SHP")
                    and "CCUV" not in n.upper()
                ),
                None,
            )
    except zipfile.BadZipFile:
        raise RuntimeError(
            f"ERROR: The file at {district_zip_path} is not a valid zip file. "
            "Please delete it from the 'data/raw' directory and run the script again to re-download."
        )

    if shp_name is None:
        raise RuntimeError("SCHOOLDISTRICTS_POLY.shp not found in schooldistricts.zip.")

    districts_gdf = gpd.read_file(f"zip://{district_zip_path}!{shp_name}")

    town_rows = []
    for _, row in districts_gdf.iterrows():
        org_code_4_digit = str(row.get("ORG4CODE", "")).zfill(4)
        org_code_8_digit = f"{org_code_4_digit}0000"
        
        members = row.get("MEMBERLIST", "")
        district_name = row.get("DISTRICT_N", "")

        if pd.notna(members) and members.strip():
            for town in [t.strip() for t in members.split(",") if t.strip()]:
                town_rows.append({"TownName": town.title(), "ORG4CODE": org_code_8_digit})
        else:
            guess = (
                district_name.replace("Public Schools", "")
                .replace("School District", "")
                .replace("Public School", "")
                .strip()
            )
            town_rows.append({"TownName": guess.title(), "ORG4CODE": org_code_8_digit})

    town_map_df = pd.DataFrame(town_rows).drop_duplicates(subset="TownName")
    
    df_edu["ORG4CODE"] = df_edu["ORG4CODE"].astype(str).str.zfill(8)
    
    print(f"[debug] Sample ORG4CODEs from DESE data: {df_edu['ORG4CODE'].unique()[:5]}")
    print(f"[debug] Sample ORG4CODEs from district shapefile (converted): {town_map_df['ORG4CODE'].unique()[:5]}")

    perf_df = pd.merge(town_map_df, df_edu[["ORG4CODE", "PerPupilExpenditure"]], on="ORG4CODE", how="left")
    print(f"[debug] Merged shapefile and DESE data. Result has {len(perf_df)} rows.")
    print(f"[debug] Found {perf_df['PerPupilExpenditure'].notna().sum()} towns with matching expenditure data.")
    perf_df["PerPupilExpenditure"] = perf_df["PerPupilExpenditure"].fillna(perf_df["PerPupilExpenditure"].median())

    scaler = MinMaxScaler()
    perf_df["EducationScore"] = scaler.fit_transform(perf_df[["PerPupilExpenditure"]])

    perf_df["TownName"] = perf_df["TownName"].apply(_normalize_town_name)

    print(f"[fetch_education_data] Completed – scored {perf_df['EducationScore'].notna().sum()} towns.")
    return perf_df[["TownName", "EducationScore"]]


def _generate_cleaned_crime_csv(raw_xls_path: Path, clean_csv_path: Path) -> str:
    """Parse the FBI UCR XLS table into a cleaned CSV next to *raw_xls_path*.
    Returns a human-readable status message detailing success or failure.
    """

    if not raw_xls_path.exists():
        return f"[crime_csv] ERROR: Raw file '{raw_xls_path}' was not found."

    try:
        raw_df = pd.read_excel(raw_xls_path, header=None, skiprows=3, engine="xlrd")
    except Exception as exc:
        return f"[crime_csv] ERROR reading Excel: {exc}"

    headers = [
        "MSA",
        "Value_Type",
        "Population",
        "Violent_Crime",
        "Murder",
        "Rape",
        "Robbery",
        "Aggravated_Assault",
        "Property_Crime",
        "Burglary",
        "Larceny_Theft",
        "Motor_Vehicle_Theft",
    ]

    processed = []
    current_msa: str | None = None

    target_descriptions = {
        "Total area actually reporting": "Reported Total",
        "Estimated total": "Estimated Total",
        "Rate per 100,000 inhabitants": "Rate per 100k",
    }

    for _, row in raw_df.iterrows():
        if pd.notna(row[0]) and pd.isna(row[1]):
            msa_name = re.sub(r"(\s+\d+|\s+M\.S\.A\..*)$", "", str(row[0])).strip()
            current_msa = msa_name
            continue

        if current_msa and pd.isna(row[0]) and pd.notna(row[1]):
            row_desc = str(row[1]).strip()
            for key, std_desc in target_descriptions.items():
                if key in row_desc:
                    processed.append(
                        {
                            "MSA": current_msa,
                            "Value_Type": std_desc,
                            "Population": row[2],
                            "Violent_Crime": row[3],
                            "Murder": row[4],
                            "Rape": row[5],
                            "Robbery": row[6],
                            "Aggravated_Assault": row[7],
                            "Property_Crime": row[8],
                            "Burglary": row[9],
                            "Larceny_Theft": row[10],
                            "Motor_Vehicle_Theft": row[11],
                        }
                    )
                    break

    if not processed:
        return "[crime_csv] ERROR: No rows extracted. XLS structure may have changed."

    clean_df = pd.DataFrame(processed, columns=headers)

    for col in headers[2:]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df.dropna(how="all", subset=headers[2:], inplace=True)

    def _std(v: str) -> str:
        if "Rate" in v:
            return "Rate per 100k"
        if "Estimated" in v:
            return "Estimated Total"
        if "Total" in v:
            return "Reported Total"
        return v

    clean_df["Value_Type"] = clean_df["Value_Type"].astype(str).apply(_std)

    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_csv_path, index=False)

    return f"[crime_csv] Success. Cleaned dataset written to '{clean_csv_path}'."


def _generate_cleaned_ma_city_csv(raw_xls_path: Path, clean_csv_path: Path) -> str:
    """Parse the Massachusetts city-level UCR XLS into a CSV.

    Returns a status message. The cleaned CSV will
    contain at least the columns City, Population, Violent_crime, Property_crime.
    """

    if not raw_xls_path.exists():
        return f"[ma_city_csv] ERROR: Raw file '{raw_xls_path}' was not found."

    try:
        raw_df = pd.read_excel(raw_xls_path, header=None, engine="xlrd")
    except Exception as exc:
        return f"[ma_city_csv] ERROR reading Excel: {exc}"

    header_candidates = raw_df[raw_df.apply(lambda r: r.astype(str).str.contains("Population", case=False).any(), axis=1)]
    if header_candidates.empty:
        return "[ma_city_csv] ERROR: Could not find header row containing 'Population'."

    header_idx = header_candidates.index[0]

    df = pd.read_excel(raw_xls_path, header=header_idx, engine="xlrd")

    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    required_cols = {"City", "Population", "Violent crime", "Property crime"}
    if not required_cols.issubset(df.columns):
        return f"[ma_city_csv] ERROR: Required columns missing ({required_cols - set(df.columns)})"

    df = df[df["City"].notna()]
    df = df[~df["City"].str.contains("Rate per 100", na=False)]

    for col in ["Population", "Violent crime", "Property crime"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Population"] > 0]

    out_cols = ["City", "Population", "Violent crime", "Property crime"]
    df[out_cols].to_csv(clean_csv_path, index=False)

    return f"[ma_city_csv] Success. Cleaned dataset written to '{clean_csv_path}'."


def fetch_crime_data() -> pd.DataFrame:
    """Load and process FBI UCR data."""
    print("[fetch_crime_data] Starting crime data acquisition…")

    ma_city_path = download_file(FBI_MA_CITY_URL, RAW_DIR / "massachusetts.xls")

    city_csv_path = RAW_DIR / "ma_city_2019_cleaned.csv"

    if not city_csv_path.exists():
        msg = _generate_cleaned_ma_city_csv(ma_city_path, city_csv_path)
        print(msg)

    if city_csv_path.exists():
        city_df = pd.read_csv(city_csv_path)

        if "CrimeRate" not in city_df.columns:
            city_df["CrimeRate"] = (
                city_df["Violent crime"] + city_df["Property crime"]
            ) / city_df["Population"] * 100_000

        city_df["TownName"] = city_df["City"].apply(_normalize_town_name)

        scaler = MinMaxScaler()
        city_df["SafetyScore"] = 1 - scaler.fit_transform(city_df[["CrimeRate"]])

        print(f"[fetch_crime_data] Successfully created SafetyScore for {len(city_df)} MA cities (Table 8).")
        return city_df[["TownName", "SafetyScore"]]

    print("[fetch_crime_data] WARN: City-level parser failed. Falling back to MSA-level data …")

    raw_xls_path = download_file(FBI_UCR_URL, RAW_DIR / "ucr2019.xls")

    clean_csv_path = RAW_DIR / "ucr2019_cleaned_dataset.csv"

    if not clean_csv_path.exists():
        print("[fetch_crime_data] Clean CSV not found. Generating from raw XLS …")
        msg = _generate_cleaned_crime_csv(raw_xls_path, clean_csv_path)
        print(msg)

    if not clean_csv_path.exists():
        raise RuntimeError("Failed to generate cleaned crime CSV; aborting crime data step.")

    df = pd.read_csv(clean_csv_path)

    df_ma = df[df["MSA"].str.contains("MA", na=False)]
    rate_rows = df_ma[df_ma["Value_Type"] == "Rate per 100k"]

    rate_rows["CrimeRate"] = (
        pd.to_numeric(rate_rows["Violent_Crime"], errors="coerce")
        + pd.to_numeric(rate_rows["Property_Crime"], errors="coerce")
    )

    def _msa_to_town(msa: str) -> str:
        if pd.isna(msa):
            return msa
        town = msa.split("-")[0].split(",")[0].strip()
        return _normalize_town_name(town)

    rate_rows["TownName"] = rate_rows["MSA"].apply(_msa_to_town)
    rate_rows = rate_rows[rate_rows["TownName"].notna()]

    scaler = MinMaxScaler()
    rate_rows["SafetyScore"] = 1 - scaler.fit_transform(rate_rows[["CrimeRate"]])

    print(f"[fetch_crime_data] Successfully created SafetyScore for {len(rate_rows)} MA MSAs (fallback).")
    return rate_rows[["TownName", "SafetyScore"]]


def build_master_dataframe(census_api_key: Optional[str] = None) -> gpd.GeoDataFrame:
    """Builds the master feature dataset by running all pipeline steps.

    This is the main entry point for the data pipeline. It orchestrates the
    loading of individual datasets, merges them, and computes final features.
    """

    towns_geo = load_towns_geo()
    acs_data = fetch_acs_data(api_key=census_api_key)
    education_data = fetch_education_data()
    crime_data = fetch_crime_data()

    df = towns_geo.copy()

    _null_report = lambda cols, tag: logger.info(
        f"[debug] Null counts after {tag}: " + ", ".join(f"{c}={n}" for c, n in df[cols].isnull().sum().items())
    )

    df = df.merge(
        acs_data,
        left_on="TownName",
        right_on="TownName",
        how="left",
        suffixes=("", "_acs"),
    )

    _null_report(["Population", "MedianIncome", "MedianHomePrice"], "ACS merge")

    df = df.merge(
        education_data,
        left_on="TownName",
        right_on="TownName",
        how="left",
        suffixes=("", "_edu"),
    )

    _null_report(["EducationScore"], "Education merge")

    missing_edu = df.loc[df["EducationScore"].isna(), "TownName"].tolist()
    if missing_edu:
        logger.info(f"[debug] EducationScore missing for {len(missing_edu)} towns: {missing_edu[:15]}{' …' if len(missing_edu) > 15 else ''}")

    df = df.merge(
        crime_data,
        left_on="TownName",
        right_on="TownName",
        how="left",
        suffixes=("", "_crime"),
    )

    _null_report(["SafetyScore"], "Crime merge")

    missing_safe = df.loc[df["SafetyScore"].isna(), "TownName"].tolist()
    if missing_safe:
        logger.info(f"[debug] SafetyScore missing for {len(missing_safe)} towns: {missing_safe[:15]}{' …' if len(missing_safe) > 15 else ''}")

    print("[build_master_dataframe] Merged all data sources.")
    print("Null counts after merge:")
    print(df[['EducationScore', 'SafetyScore', 'MedianHomePrice', 'Population']].isnull().sum())


    df["PopulationDensity"] = df["Population"] / df["SQ_MILES"]

    df = _ensure_lat_lon(df)
    df = add_walk_scores(df)
    df = add_amenities_score(df)

    def _multiple_impute_scores(df_in: pd.DataFrame, n_imputations: int = 20) -> pd.DataFrame:
        target_cols = ["EducationScore", "SafetyScore"]
        helper_cols = [
            "PopulationDensity",
            "MedianIncome",
            "MedianHomePrice",
            "WalkScore",
            "AmenitiesScore",
        ]

        impute_cols = target_cols + helper_cols
        X = df_in[impute_cols]

        imputed_samples = []
        base_imputer = IterativeImputer(
            sample_posterior=True,
            max_iter=15,
            random_state=0,
            n_nearest_features=None,
        )

        for seed in range(n_imputations):
            imputer = base_imputer
            imputer.random_state = seed
            imputed = imputer.fit_transform(X)
            imputed_samples.append(imputed[:, : len(target_cols)])  # only target cols

        imputed_stack = np.stack(imputed_samples, axis=2)  # shape (n_samples, 2, n_imputations)
        means = imputed_stack.mean(axis=2)
        stds = imputed_stack.std(axis=2)

        for idx, col in enumerate(target_cols):
            mean_vals = means[:, idx]
            std_vals = stds[:, idx]

            imputed_flag_col = f"{col}Imputed"
            imputed_std_col = f"{col}ImputedStd"

            df_in[imputed_flag_col] = df_in[col].isna()

            df_in.loc[df_in[col].isna(), col] = mean_vals[df_in[col].isna()]

            df_in[imputed_std_col] = std_vals

        logger.info("[imputation] Completed multiple imputation for missing scores.")
        return df_in

    df = _multiple_impute_scores(df)

    _null_report(["EducationScore", "SafetyScore"], "Post-imputation")

    final_fill_values = {
        'Population': 0,
        'MedianIncome': 0,
        'MedianHomePrice': 0,
        'WalkScore': 0,
        'AmenitiesScore': 0,
        'PopulationDensity': 0
    }
    df = df.fillna(value=final_fill_values)

    print("[build_master_dataframe] Pipeline complete.")
    return df


def add_normalised_columns(df: pd.DataFrame, cols_to_norm: List[str]) -> pd.DataFrame:
    """Normalise specified columns using MinMaxScaler. Handles NaNs."""
    for col in cols_to_norm:
        scaler = MinMaxScaler()
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            
            df[f"{col}_norm"] = scaler.fit_transform(df[[col]])
        else:
            print(f"[normalise] WARN: Column '{col}' not found or not numeric. Skipping.")
    return df


def _parse_args():
    p = argparse.ArgumentParser(description="Build SmartTownMatch master dataset.")
    p.add_argument("--output", default=str(PROCESSED_DIR / "master_town_data.parquet"), help="Output parquet path")
    p.add_argument("--census-key", dest="api_key", help="Census API key (or set CENSUS_API_KEY env var)")
    return p.parse_args()


def main():
    """Main ETL script entry point."""
    args = _parse_args()
    print("Starting SmartTownMatch data pipeline…")

    master_df = build_master_dataframe(census_api_key=os.getenv("CENSUS_API_KEY"))

    numeric_features = [
        "PopulationDensity",
        "MedianIncome",
        "MedianHomePrice",
        "SafetyScore",
        "EducationScore",
        "WalkScore",
        "AmenitiesScore",
        "TransitScore",
        "BikeScore",
    ]
    master_df = add_normalised_columns(master_df, numeric_features)

    output_path = Path(args.output)
    _ensure_dir(output_path)
    print(f"Saving final master dataset to {output_path}…")
    if "geometry" in master_df.columns:
        master_df.to_parquet(output_path)
    else:
        master_df.to_csv(output_path)

    print("Done.")


if __name__ == "__main__":
    main() 