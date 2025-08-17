# bu_region_bucketizer.py
# ------------------------------------------------------------
# Schneider Electric — BU & Region Bucketizer (standalone)
# ------------------------------------------------------------
# Exposes:
#   classify_bu(text, hint_code=None) -> (bu_code, bu_name)
#   bucketize_region(country, state=None, city=None, text_hint=None) -> region
#   classify_bu_series(df, text_cols) -> pd.Series of BU code
#   bucketize_region_series(df, country_col='Country', state_col='State',
#                           city_col='City', text_cols=None) -> pd.Series
#   enrich_with_bu_region(df, text_cols, country_col='Country',
#                         state_col='State', city_col='City') -> df with bu_code, bu_name, region
#
# Regions returned: "North", "East", "South", "West", "NC", "Others"
#   - "NC": neighboring (included) countries only: Nepal, Bhutan, Bangladesh, Sri Lanka, Maldives
#   - Explicit exclusions (Pakistan, China, Afghanistan, Myanmar) → "Others"
#   - Unknown/ambiguous → "Others"
#
# BU codes (retained): PPIBS, PSIBS, IDIBS, SPIBS (+ BMS, H&D, A2E, Solar, OTHER)
# Common aliases normalized: PP→PPIBS, PS→PSIBS, IA→IDIBS, SP→SPIBS, HD→H&D, LV→PPIBS, MV→PSIBS
# ------------------------------------------------------------

from __future__ import annotations
import re
from typing import Iterable, Optional, Tuple, Dict
import pandas as pd

# ------------------------- BU definitions (RETAINED) -------------------------
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

# Accept common shorthands but normalize to retained codes
_ALIAS_BU = {
    "PPIBS": "PPIBS", "PSIBS": "PSIBS", "IDIBS": "IDIBS", "SPIBS": "SPIBS",
    "BMS": "BMS", "H&D": "H&D", "A2E": "A2E", "SOLAR": "Solar", "OTHER": "OTHER",
    # Shorthands:
    "PP": "PPIBS", "PS": "PSIBS", "IA": "IDIBS", "SP": "SPIBS", "HD": "H&D",
    "LV": "PPIBS", "MV": "PSIBS",
}

# ------------------------- BU keyword rules (ordered) -------------------------
# First match wins; crafted to reduce cross-BU false positives.
_BU_RULES = [
    # Secure Power (data centers & critical power)
    ("SPIBS",
     r"\b("
     r"apc|smart[-\s]?ups|easy\s*ups|symmetra(?:\s*lx)?|"
     r"galaxy(?:\s*(?:vs|vl|vx|vm))?|"
     r"netshelter|netbotz|"
     r"rack\s*(?:pdu|ats)|rpdus?|pdu[s]?|"
     r"uniflair|inrow|row\s*cooling|compressor|condensor|pump|lib|chiller|"
     r"(?:precision\s*)?cooling|crac|crah|ups|outdoor|indoor|pipe|LMU|danfoss|uniflair|pcb|pcba|board|galaxy|ups|phe|ahu|leak|"
     r"battery(?:\s*bank)?|"
     r"micro\s*data\s*center|"
     r"ecostruxure\s*it(?:\s*(?:expert|advisor|gateway))?|dcim"
     r")\b"),

    # Building Management Systems
    ("BMS",
     r"\b("
     r"bms|building\s*management\s*system|"
     r"ecostruxure\s*building\s*(?:operation|expert|advisor)|ebo\b|"
     r"spacelogic|smartx|"
     r"tac\s*(?:vista)?|continuum|"
     r"room\s*controller|"
     r"(?:hvac|lighting|access)\s*control|smart\s*building|"
     r"bacnet|knx|lon\b"
     r")\b"),

    # Industrial Automation & Control
    ("IDIBS",
     r"\b("
     r"modicon|m2(?:21|41|51|62)|m340|m580|quantum|premium|"
     r"altivar|atv\d+|lexium|pacdrive|harmony|magelis|"
     r"(?:plc|pac|scada|dcs)\b|"
     r"triconex|foxboro|"
     r"safety\s*instrumented\s*system|sis\b|iiot|"
     r"ecostruxure\s*(?:machine|plant|process)\b|"
     r"(?:control\s*expert|unity\s*pro|somachine|vijeo\s*designer)|"
     r"scadapack|remoteconnect|telemetry\b|"
     r"telemecanique|osisense|zelio\s*logic|ats"
     r")\b"),

    # Medium Voltage / Grid Automation
    ("PSIBS",
     r"\b("
     r"medium[\s-]?voltage|mv\b|"
     r"airset|rm6|sm6|premset|pix\b|"
     r"gis\b|ais\b|sf6\b|"
     r"(?:easergy(?:\s*p[35])?|micom|vamp)\b|t300\b|hwx|pix|hvx|trihal|oil|transformer|easergy|vaccum|relay|sensor"
     r"ring\s*main\s*unit|rmu[s]?|"
     r"recloser|sectionalizer|vcb|vcu|breaker|eto|ecofit|"
     r"(?:vcb|vacuum\s*circuit\s*breaker)\b|"
     r"(?:adms|derms)\b|"
     r"substation\s*automation|grid\s*(?:monitoring|control|automation)"
     r")\b"),

    # Low Voltage / Power Products & Systems
    ("PPIBS",
     r"\b("
     r"low[\s-]?voltage|lv\b|power\s*products?|"
     r"compact\s*nsx(?:m)?|nsx(?:m)?\b|masterpact(?:\s*mtz)?|acti9|"
     r"(?:mc|m|r)c(?:cb|bo)\b|mcb|mccb|rcd|rccb|rcbo|mpcb|acb\b|"
     r"prisma(?:set)?|okken|blokset|canalis|"
     r"tesys|zelio(?!\s*logic)|powerlogic|ion\s*\d+|accusine|pm\d{3,4}|"
     r"panelboard|switchboard|"
     r"capacitor\s*bank|apfc|power\s*factor|"
     r"changeover|ats\b|isolator|switch\s*disconnector|pcc|mcc|nsx"
     r"easypact(?:\s*ezc)?"
     r")\b"),

    # Home & Distribution (Residential/Small business)
    ("H&D",
     r"\b("
     r"wiser(?!\s*energy\s*center)|clipsal|avataron|vivace|livia|zencelo|opale|neo|ulti|unica|merten|"
     r"(?:wiring\s*device|switch(?:es)?\s*&?\s*sockets?)|"
     r"residential|home\s*automation|"
     r"arc\s*fault|afdd|surge\s*protection|smart\s*panel|resi9"
     r")\b"),

    # Access to Energy
    ("A2E",
     r"\b("
     r"access\s*to\s*energy|a2e|homaya|villaya|mobiya|"
     r"rural\s*electrification|off[-\s]?grid(?:\s*solar)?|mini[-\s]?grid|pay[-\s]?as[-\s]?you[-\s]?go"
     r")\b"),

    # Solar & Storage
    ("Solar",
     r"\b("
     r"solar\s*(?:inverter|pv|string|hybrid)|mppt|charge\s*controller|"
     r"conext|xw\s*pro|sw\s*inverter|"
     r"solar\s*combiner|"
     r"bess|battery\s*(?:system|storage)"
     r")\b"),
]

# Recompile after edits
_BU_COMPILED = [(code, re.compile(pat, re.I)) for code, pat in _BU_RULES]

# ------------------------- Normalization helpers -------------------------
def _norm(x: Optional[str]) -> str:
    """Lowercase & collapse whitespace/punctuations to single spaces for robust matching."""
    if not x:
        return ""
    s = re.sub(r"[\s\.\-_,/&]+", " ", str(x).strip().lower())
    return re.sub(r"\s+", " ", s)

def _normalize_bu_code(code: Optional[str]) -> str:
    if not code:
        return "OTHER"
    code_up = _norm(code).upper()
    return _ALIAS_BU.get(code_up, code_up if code_up in BU_NAME_BY_CODE else "OTHER")

# ------------------------- BU classifier -------------------------
def classify_bu(text: Optional[str], hint_code: Optional[str] = None) -> Tuple[str, str]:
    """
    Return (bu_code, bu_name).
    Priority:
      1) if hint_code provided -> normalized & returned
      2) else regex match on text (first match)
      3) else OTHER
    """
    if hint_code:
        code = _normalize_bu_code(hint_code)
        return code, BU_NAME_BY_CODE.get(code, BU_NAME_BY_CODE["OTHER"])

    s = _norm(text)
    for code, rx in _BU_COMPILED:
        if rx.search(s):
            code = _ALIAS_BU.get(code, code)
            return code, BU_NAME_BY_CODE.get(code, BU_NAME_BY_CODE["OTHER"])
    return "OTHER", BU_NAME_BY_CODE["OTHER"]

# ------------------------- Region (India + NC) -------------------------
# State/UT → Region
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

# Aliases/old names to canonical state name
ALIASES_STATE = {
    "orissa":"odisha",
    "uttaranchal":"uttarakhand",
    "pondicherry":"puducherry",
    "a&n islands":"andaman and nicobar islands",
    "andaman & nicobar":"andaman and nicobar islands",
    "andaman and nicobar":"andaman and nicobar islands",
    "nct of delhi":"delhi",
    "daman and diu":"dadra and nagar haveli and daman and diu",
    "dadra and nagar haveli":"dadra and nagar haveli and daman and diu",
    "dnhdd":"dadra and nagar haveli and daman and diu",
    "dnh&dd":"dadra and nagar haveli and daman and diu",
}

# City → State (extend as needed)
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
    "panaji":"goa","vasco da gama":"goa",
    "daman":"dadra and nagar haveli and daman and diu","silvassa":"dadra and nagar haveli and daman and diu",
    # South
    "chennai":"tamil nadu","coimbatore":"tamil nadu","madurai":"tamil nadu",
    "bengaluru":"karnataka","bangalore":"karnataka","mysuru":"karnataka","mysore":"karnataka",
    "mangaluru":"karnataka","mangalore":"karnataka","hyderabad":"telangana","warangal":"telangana",
    "visakhapatnam":"andhra pradesh","vizag":"andhra pradesh","vijayawada":"andhra pradesh","tirupati":"andhra pradesh",
    "kochi":"kerala","cochin":"kerala","thiruvananthapuram":"kerala","trivandrum":"kerala",
    "puducherry":"puducherry","pondicherry":"puducherry","kavaratti":"lakshadweep",
    # East
    "kolkata":"west bengal","howrah":"west bengal","siliguri":"west bengal","durgapur":"west bengal","asansol":"west bengal",
    "bhubaneswar":"odisha","cuttack":"odisha","rourkela":"odisha",
    "patna":"bihar","gaya":"bihar",
    "ranchi":"jharkhand","jamshedpur":"jharkhand","dhanbad":"jharkhand",
    "guwahati":"assam","dispur":"assam","imphal":"manipur","shillong":"meghalaya","aizawl":"mizoram",
    "kohima":"nagaland","agartala":"tripura","gangtok":"sikkim","itanagar":"arunachal pradesh",
    "port blair":"andaman and nicobar islands",
}

# Neighboring countries to mark as NC (others become "Others")
NC_INCLUDE = {"nepal","bhutan","bangladesh","sri lanka","srilanka","maldives"}
# Explicit exclusions → Others
NC_EXCLUDE = {"pakistan","china","people's republic of china","pr china","prc","afghanistan","myanmar","burma"}

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

    # Non-India handling
    if c != "india":
        if c in NC_INCLUDE:
            return "NC"
        # Exclusions & all others → Others
        return "Others"

    # India → prefer state
    if st and st in REGION_BY_STATE:
        return REGION_BY_STATE[st]

    # Fallback to city
    if ct and ct in CITY_TO_STATE:
        return REGION_BY_STATE.get(CITY_TO_STATE[ct], "Others")

    # As a last resort, sniff city in free text
    if text_hint:
        s = _norm(text_hint)
        for ci, st_ in CITY_TO_STATE.items():
            if f" {ci} " in f" {s} ":
                return REGION_BY_STATE.get(st_, "Others")

    return "Others"

# ------------------------- Pandas helpers -------------------------
def classify_bu_series(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.Series:
    """
    Vectorized BU classification over DataFrame.
    text_cols: columns whose text will be concatenated to detect BU (e.g., ["Issue","Customer"])
    """
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
    """
    Vectorized Region bucketing over DataFrame.
    Optionally uses text_cols as a hint if country/state/city are missing.
    """
    hint = None
    if text_cols:
        cols = [c for c in text_cols if c in df.columns]
        if cols:
            hint = df[cols].astype(str).agg(" ".join, axis=1)

    # Use iloc for position-based indexing to avoid surprises with non-range indexes
    regions = []
    n = len(df)
    for i in range(n):
        country = df[country_col].iloc[i] if country_col in df.columns else None
        state   = df[state_col].iloc[i]   if state_col   in df.columns else None
        city    = df[city_col].iloc[i]    if city_col    in df.columns else None
        txt     = hint.iloc[i] if isinstance(hint, pd.Series) else None
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

    # BU (vectorized)
    cols = [c for c in text_cols if c in out.columns]
    if cols:
        txt = out[cols].astype(str).agg(" ".join, axis=1)
    else:
        txt = pd.Series([""] * len(out), index=out.index)

    bu_pairs = txt.map(classify_bu)
    out["bu_code"] = bu_pairs.map(lambda x: x[0]).fillna("OTHER")
    out["bu_name"] = bu_pairs.map(lambda x: x[1]).fillna(BU_NAME_BY_CODE["OTHER"])

    # Region
    out["region"] = bucketize_region_series(
        out, country_col=country_col, state_col=state_col, city_col=city_col, text_cols=cols
    )
    return out

# ------------------------------ Demo ------------------------------
if __name__ == "__main__":
    demo = pd.DataFrame({
        "Customer": ["A","B","C","D","E","F","G","H","I","J"],
        "Issue": [
            "ComPacT NSX breaker tripped – need replacement",
            "AirSeT RMU SF6-free MV – relay nuisance trip",
            "APC Galaxy VS UPS alarm in DC; check EcoStruxure IT DCIM",
            "Modicon M580 PLC IO fault with Altivar ATV320 drive",
            "EcoStruxure Building Operation (EBO) alarm in AHU",
            "Wiser smart home dimmer issue in apartment",
            "Conext XW Pro hybrid solar inverter failure",
            "Villaya microgrid service request for rural electrification",
            "Okken LV switchboard with Canalis busway extension",
            "MiCOM relay nuisance trip in substation – check Easergy P3"
        ],
        "Country": ["India","India","India","India","India","India","India","Nepal","India","India"],
        "State":   ["Maharashtra","Uttar Pradesh","Karnataka","Tamil Nadu","Delhi","Kerala","Odisha","", "Gujarat","Rajasthan"],
        "City":    ["Mumbai","Noida","Bengaluru","Chennai","New Delhi","Kochi","Bhubaneswar","Kathmandu","Ahmedabad","Jaipur"]
    })
    enriched = enrich_with_bu_region(
        demo, text_cols=["Issue","Customer"], country_col="Country", state_col="State", city_col="City"
    )
    print(enriched[["Issue","bu_code","bu_name","region"]])
