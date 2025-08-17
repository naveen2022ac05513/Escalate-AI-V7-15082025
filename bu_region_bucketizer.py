# bu_region_bucketizer.py
# ------------------------------------------------------------
# Schneider Electric — BU & Region Bucketizer (standalone)
# ------------------------------------------------------------
# Exposes:
#   classify_bu(text) -> (bu_code, bu_name)
#   bucketize_region(country, state=None, city=None, text_hint=None) -> region
#   classify_bu_series(df, text_cols) -> pd.Series of BU code
#   bucketize_region_series(df, country_col='Country', state_col='State', city_col='City', text_cols=None) -> pd.Series
#   enrich_with_bu_region(df, text_cols, country_col='Country', state_col='State', city_col='City') -> df with bu_code, bu_name, region
#
# Region categories returned: "North", "East", "South", "West", "NC", "Others"
# - "NC": neighboring (included) countries only: Nepal, Bhutan, Bangladesh, Sri Lanka, Maldives
# - exclude: Pakistan, China, Afghanistan, Myanmar → "Others"
# - Everything non-India that isn’t NC → "Others"
#
# BU codes:
#   SPIBS (Secure Power), PPIBS (Low Voltage), PSIBS (Medium Voltage),
#   IDIBS (Industrial Automation), BMS, H&D, A2E, Solar, OTHER
# ------------------------------------------------------------

from __future__ import annotations
import re
from typing import Iterable, Optional, Tuple, Dict
import pandas as pd

# ------------------------- BU definitions -------------------------
BU_MAP = [
    ("PPIBS", "Low Voltage Products & Systems"),
    ("PSIBS", "Medium Voltage Distribution & Grid Automation"),
    ("IDIBS", "Industrial Automation & Control"),
    ("BMS",   "Building Automation & Control"),
    ("SPIBS", "Critical Power, Cooling & Racks"),
    ("H&D",   "Residential & Small Business"),
    ("A2E",   "Access to Energy"),
    ("Solar", "Solar & Energy Storage"),
    ("OTHER", "Other / Unclassified"),
]
BU_NAME_BY_CODE = {c: n for c, n in BU_MAP}
_ALIAS_BU = {"HD": "H&D", "H&D": "H&D"}  # keep H&D as-is

# BU keyword rules (ordered, most specific first) — expanded vocabulary
_BU_RULES = [
    # Secure Power (SPIBS)
    ("SPIBS", r"\b("
              r"apc|smart[-\s]?ups|easy\s*ups|symmetra|galaxy\b|"
              r"netshelter|netbotz|rpdus?|rack\s*pdu|"
              r"ecostruxure\s*it|dcim|uniflair|inrow|row[-\s]?cooling|precision\s*cooling|"
              r"micro\s*data\s*center|mdc|cooling|chiller|ahu|compressor|condenser|"
              r"battery|lithium|li[-\s]?ion|ups(?:\b|s\b)|pdu\b"
              r")\b"),

    # Building Management (BMS)
    ("BMS", r"\b("
            r"bms\b|building\s*management\s*system|ecostruxure\s*building\s*(operation|expert)|"
            r"spacelogic|smartx|tac\s*(vista)?|continuum|room\s*controller|"
            r"(hvac|lighting|access)\s*control|vav\b|ahu\b|smart\s*building"
            r")\b"),

    # Industrial Automation (IDIBS)
    ("IDIBS", r"\b("
              r"modicon|m221|m241|m251|m262|m340|m580|quantum|premium|"
              r"altivar|atv\d+|lexium|pacdrive|harmony|magelis|"
              r"(plc|pac|scada|dcs)\b|triconex|foxboro|"
              r"safety\s*instrumented\s*system|sis\b|iiot|edge\s*box|"
              r"aveva|citect\b|plant\sscada|system\s*platform"
              r")\b"),

    # Medium Voltage / Grid (PSIBS)
    ("PSIBS", r"\b("
              r"medium[\s-]?voltage|mv\b|"
              r"airset|rm6|sm6|premset|pix\b|gis\b|"
              r"hwx|hvx|vmx|"
              r"transformer|trihal|easypact|micom|easergy|vamp|"
              r"relay|protection|"
              r"easergy\s*p[35]\b|t300\b|ring\s*main\s*unit|rmu[s]?|"
              r"(adms|derms)\b|substation\s*automation|grid\s*(monitoring|control|automation)"
              r")\b"),

    # Low Voltage / Power Products (PPIBS)
    ("PPIBS", r"\b("
              r"low[\s-]?voltage|lv\b|power\s*products?|"
              r"compact\s*nsx[m]?|masterpact(\s*mtz)?|acti9|prisma(?:set)?|canalis|"
              r"tesys|zelio|powerlogic|ion\s*\d+|accusine|pm\d{3,4}|"
              r"panelboard|switchboard|capacitor\s*bank|power\s*factor\s*correction|pfc|"
              r"mccb|acb|mcb|rcbo|contactor|overload\s*relay|busway|busbar"
              r")\b"),

    # Home & Distribution (H&D)
    ("H&D", r"\b("
            r"wiser(?!\s*energy\s*center)|clipsal|avataron|vivace|livia|"
            r"(wiring\s*device|switch(?:es)?\s*&?\s*sockets?)|"
            r"residential|home\s*automation|smart\s*home|"
            r"arc\s*fault|afdd|surge\s*protection|smart\s*panel|square\s*d"
            r")\b"),

    # Access to Energy (A2E)
    ("A2E", r"\b("
            r"access\s*to\s*energy|a2e|homaya|villaya|mobiya|"
            r"rural\s*electrification|off[-\s]?grid(\s*solar)?|mini[-\s]?grid|"
            r"pay[-\s]?as[-\s]?you[-\s]?go"
            r")\b"),

    # Solar / Storage
    ("Solar", r"\b("
              r"solar\s*(inverter|pv|string|hybrid)|mppt|charge\s*controller|"
              r"conext|xw\s*pro|sw\s*inverter|solar\s*combiner|"
              r"bess|battery\s*(system|storage)|ess|hybrid\s*inverter"
              r")\b"),
]
_BU_COMPILED = [(code, re.compile(pat, re.I)) for code, pat in _BU_RULES]

def _norm(x: Optional[str]) -> str:
    if not x: return ""
    s = re.sub(r"[\s\.\-_,&/]+", " ", str(x).strip().lower())
    return re.sub(r"\s+", " ", s)

def classify_bu(text: Optional[str]) -> Tuple[str, str]:
    """
    Return (bu_code, bu_name) for the given text.
    """
    s = _norm(text)
    for code, rx in _BU_COMPILED:
        if rx.search(s):
            code = _ALIAS_BU.get(code, code)
            return code, BU_NAME_BY_CODE.get(code, "Other / Unclassified")
    return "OTHER", BU_NAME_BY_CODE["OTHER"]

# ---------------------- Region (India + NC) -----------------------
REGION_BY_STATE: Dict[str, str] = {
    # North
    "jammu and kashmir":"North","ladakh":"North","himachal pradesh":"North","punjab":"North",
    "chandigarh":"North","haryana":"North","delhi":"North","uttarakhand":"North",
    "rajasthan":"North","uttar pradesh":"North",
    # West
    "gujarat":"West","maharashtra":"West","goa":"West","madhya pradesh":"West",
    "dadra and nagar haveli and daman and diu":"West",
    # South
    "karnataka":"South","kerala":"South","tamil nadu":"South","telangana":"South",
    "andhra pradesh":"South","puducherry":"South","lakshadweep":"South",
    # East (incl. NE + A&N)
    "bihar":"East","jharkhand":"East","odisha":"East","west bengal":"East","sikkim":"East",
    "assam":"East","arunachal pradesh":"East","manipur":"East","meghalaya":"East",
    "mizoram":"East","nagaland":"East","tripura":"East","chhattisgarh":"East",
    "andaman and nicobar islands":"East",
}
ALIASES_STATE = {
    "orissa":"odisha","uttaranchal":"uttarakhand","pondicherry":"puducherry",
    "a&n islands":"andaman and nicobar islands","andaman & nicobar":"andaman and nicobar islands",
    "andaman and nicobar":"andaman and nicobar islands",
    "daman and diu":"dadra and nagar haveli and daman and diu",
    "dadra and nagar haveli":"dadra and nagar haveli and daman and diu",
    "nct of delhi":"delhi",
}
CITY_TO_STATE = {
    # North
    "delhi":"delhi","new delhi":"delhi","gurgaon":"haryana","gurugram":"haryana","faridabad":"haryana",
    "noida":"uttar pradesh","ghaziabad":"uttar pradesh","lucknow":"uttar pradesh","kanpur":"uttar pradesh",
    "jaipur":"rajasthan","udaipur":"rajasthan","jodhpur":"rajasthan","chandigarh":"chandigarh",
    "amritsar":"punjab","ludhiana":"punjab","shimla":"himachal pradesh","dehradun":"uttarakhand",
    # West
    "mumbai":"maharashtra","navi mumbai":"maharashtra","thane":"maharashtra","pune":"maharashtra",
    "nagpur":"maharashtra","nashik":"maharashtra",
    "ahmedabad":"gujarat","surat":"gujarat","vadodara":"gujarat","baroda":"gujarat","rajkot":"gujarat",
    "indore":"madhya pradesh","bhopal":"madhya pradesh","ujjain":"madhya pradesh",
    "panaji":"goa","vasco da gama":"goa","daman":"dadra and nagar haveli and daman and diu","silvassa":"dadra and nagar haveli and daman and diu",
    # South
    "chennai":"tamil nadu","coimbatore":"tamil nadu","madurai":"tamil nadu",
    "bengaluru":"karnataka","bangalore":"karnataka","mysuru":"karnataka","mysore":"karnataka",
    "mangaluru":"karnataka","mangalore":"karnataka","hyderabad":"telangana","warangal":"telangana",
    "visakhapatnam":"andhra pradesh","vizag":"andhra pradesh","vijayawada":"andhra pradesh","tirupati":"andhra pradesh",
    "kochi":"kerala","cochin":"kerala","thiruvananthapuram":"kerala","trivandrum":"kerala",
    "puducherry":"puducherry","pondicherry":"puducherry","kavaratti":"lakshadweep",
    # East
    "kolkata":"west bengal","howrah":"west bengal","siliguri":"west bengal","durgapur":"west bengal","asansol":"west bengal",
    "bhubaneswar":"odisha","cuttack":"odisha","rourkela":"odisha","patna":"bihar","gaya":"bihar",
    "ranchi":"jharkhand","jamshedpur":"jharkhand","dhanbad":"jharkhand",
    "guwahati":"assam","dispur":"assam","imphal":"manipur","shillong":"meghalaya","aizawl":"mizoram",
    "kohima":"nagaland","agartala":"tripura","gangtok":"sikkim","itanagar":"arunachal pradesh",
    "port blair":"andaman and nicobar islands",
}

# Neighboring countries to mark as NC (others become "Others")
NC_INCLUDE = {"nepal","bhutan","bangladesh","sri lanka","srilanka","maldives"}
NC_EXCLUDE = {"pakistan","china","pr china","people's republic of china","afghanistan","myanmar","burma"}

def _canon_state(s: str) -> str:
    s = _norm(s)
    return ALIASES_STATE.get(s, s)

def bucketize_region(
    country: Optional[str],
    state: Optional[str] = None,
    city: Optional[str] = None,
    text_hint: Optional[str] = None
) -> str:
    """
    Return one of: 'North' | 'East' | 'South' | 'West' | 'NC' | 'Others'
    """
    c = _norm(country)
    st = _canon_state(state) if state else ""
    ct = _norm(city)

    # Assume India if country missing but state/city exists
    if not c and (st or ct):
        c = "india"

    if not c:
        return "Others"

    if c != "india":
        if c in NC_INCLUDE:
            return "NC"
        return "Others"

    if st and st in REGION_BY_STATE:
        return REGION_BY_STATE[st]

    if ct and ct in CITY_TO_STATE:
        return REGION_BY_STATE.get(CITY_TO_STATE[ct], "Others")

    if text_hint:
        s = _norm(text_hint)
        for ci, st_ in CITY_TO_STATE.items():
            if f" {ci} " in f" {s} ":
                return REGION_BY_STATE.get(st_, "Others")

    return "Others"

# ------------------------- Pandas helpers -------------------------
def classify_bu_series(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.Series:
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        return pd.Series(["OTHER"] * len(df), index=df.index, name="bu_code")
    text = df[cols].astype(str).agg(" ".join, axis=1)
    return text.map(lambda t: classify_bu(t)[0]).rename("bu_code")

def bucketize_region_series(
    df: pd.DataFrame,
    country_col: str = "Country",
    state_col: str = "State",
    city_col: str = "City",
    text_cols: Optional[Iterable[str]] = None
) -> pd.Series:
    hint = None
    if text_cols:
        cols = [c for c in text_cols if c in df.columns]
        if cols:
            hint = df[cols].astype(str).agg(" ".join, axis=1)
    regions = []
    for i in range(len(df)):
        country = df[country_col][i] if country_col in df.columns else None
        state   = df[state_col][i]   if state_col   in df.columns else None
        city    = df[city_col][i]    if city_col    in df.columns else None
        txt     = hint[i] if isinstance(hint, pd.Series) else None
        regions.append(bucketize_region(country, state, city, text_hint=txt))
    return pd.Series(regions, index=df.index, name="region")

def enrich_with_bu_region(
    df: pd.DataFrame,
    text_cols: Iterable[str],
    country_col: str = "Country",
    state_col: str = "State",
    city_col: str = "City"
) -> pd.DataFrame:
    """
    Returns a copy of df with: bu_code, bu_name, region.
    """
    out = df.copy()
    # BU
    cols = [c for c in text_cols if c in out.columns]
    txt = out[cols].astype(str).agg(" ".join, axis=1) if cols else pd.Series([""]*len(out), index=out.index)
    bu_pairs = txt.map(classify_bu)
    out["bu_code"] = bu_pairs.map(lambda x: x[0]).fillna("OTHER")
    out["bu_name"] = bu_pairs.map(lambda x: x[1]).fillna(BU_NAME_BY_CODE["OTHER"])
    # Region
    out["region"] = bucketize_region_series(out, country_col=country_col, state_col=state_col, city_col=city_col, text_cols=cols)
    return out

# ------------------------------ Demo ------------------------------
if __name__ == "__main__":
    demo = pd.DataFrame({
        "Customer": ["A","B","C","D","E","F","G","H"],
        "Issue": [
            "ComPacT NSX breaker tripped – need replacement",
            "AirSeT RMU gas free MV – relay nuisance trip",
            "APC Galaxy UPS alarm in DC; check EcoStruxure IT",
            "Modicon M580 PLC IO fault with Altivar drive",
            "EcoStruxure Building Operation alarm in AHU",
            "Wiser smart home dimmer issue in apartment",
            "Conext XW Pro hybrid solar inverter failure",
            "Villaya microgrid service request"
        ],
        "Country": ["India","India","India","India","India","India","India","Nepal"],
        "State":   ["Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Delhi","Kerala","Odisha",""],
        "City":    ["Mumbai","Noida","Bengaluru","Chennai","New Delhi","Kochi","Bhubaneswar","Kathmandu"]
    })
    enriched = enrich_with_bu_region(demo, text_cols=["Issue"], country_col="Country", state_col="State", city_col="City")
    print(enriched[["Issue","bu_code","bu_name","region"]])
