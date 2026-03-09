from __future__ import annotations

"""
Kila Int -- Message parser / NLP layer
=======================================
Extracts structured intelligence from raw Telegram messages using
regex and keyword matching (no heavy NLP dependencies):
  - Event type classification
  - Location extraction (pattern-based)
  - Entity extraction (pattern-based)
  - Basic sentiment scoring
  - Keyword tagging
"""

import re

# ---------------------------------------------------------------------------
# Event type classification -- keyword patterns
# ---------------------------------------------------------------------------
EVENT_PATTERNS: dict[str, list[str]] = {
    "airstrike": [
        r"\bair\s*strike", r"\baerial\s+(attack|assault|bombing)",
        r"\bbombing\s+raid", r"\bsorties?\b", r"\bair\s+raid",
        r"\bstrikes?\s+(against|on|in|at)\b", r"\bextensive\s+strikes",
        r"\bjets?\s+over\b", r"\bsquadron\s+of\b.*\bjets?\b",
        r"\bbombed\b", r"\bbombing\b", r"\baerial\s+campaign",
        r"\bwar\s*planes?\b", r"\bf[\-\s]?(?:16|15|35)\b",
    ],
    "shelling": [
        r"\bshell(ing|ed|s)\b", r"\bartillery", r"\bmortar",
        r"\bbombard", r"\bnavy\s+shell", r"\bnaval\s+(fire|bombard|shell)",
        r"\bhowitzer", r"\btank\s+fire", r"\btank\s+shell",
    ],
    "explosion": [
        r"\bexplosion", r"\bblast\b", r"\bdetonat", r"\bcar\s*bomb",
        r"\bIED\b", r"\bVBIED\b", r"\bbooby[\-\s]?trap",
        r"\bimpact[s]?\b", r"\bcluster\s+(munition|warhead|bomblet|fragment)",
        r"\bfire\b.*\bafter\b.*\b(impact|strike)", r"\bsmoke\b.*\bskyline",
        r"\bcrater\b",
    ],
    "missile": [
        r"\bmissile[s]?\b", r"\brocket[s]?\b", r"\bballistic",
        r"\bcruise\s+missile", r"\bhypersonic",
        r"\blaunch(ed|es|ing)?\s+(rocket|missile|drone|UAV|at\b|toward|against)",
        r"\bintercept(ed|ion|or|s)?\b", r"\biron\s+dome",
        r"\bpatriot\b", r"\bair\s+defens", r"\banti[\-\s]?air",
        r"\bSAM\b", r"\bS[\-\s]?[234]00\b",
    ],
    "drone": [
        r"\bdrone[s]?\b", r"\bUAV[s]?\b", r"\bUAS\b",
        r"\bone[\-\s]?way\s+(attack\s+)?drone",r"\bkamikaze\s+drone",
        r"\bshahed\b", r"\borbiter\b", r"\bloitering\s+munition",
        r"\bdrones?\s+(are\s+)?target", r"\bdrones?\s+(hit|struck|over)\b",
    ],
    "ground_op": [
        r"\bground\s+(op|operation|offensive|invasion|incursion|assault)",
        r"\binfantry\b", r"\btroops?\s+(enter|cross|advance|storm|raid)",
        r"\braid(ed|ing|s)?\b", r"\bspecial\s+forces?\b",
        r"\bcommando\b", r"\bhelicopter[s]?\s+(land|deploy)",
        r"\blanded\b.*\bhelicopter", r"\bengaging\s+(the\s+)?(\w+\s+)?forces",
    ],
    "military_movement": [
        r"\btroop\s+movement", r"\bdeployment", r"\breinforcement",
        r"\bconvoy\b", r"\bmobiliz", r"\badvance[ds]?\b.*front",
        r"\bbuildup\b", r"\bamass", r"\bposition(ed|ing)\b",
        r"\bfleet\b", r"\bcarrier\b.*\b(group|strike)",
        r"\bnaval\s+(task|strike|battle)\b",
    ],
    "casualty": [
        r"\bcasualti", r"\bkilled\b", r"\bdead\b", r"\bwounded\b",
        r"\binjured\b", r"\bfatalit", r"\bdeath\s+toll",
        r"\bmartyr", r"\bdied\b", r"\blives?\s+lost",
        r"\bbodies\b", r"\bmass\s+casualt",
    ],
    "clash": [
        r"\bclash", r"\bfirefight", r"\bskirmish", r"\bgunfight",
        r"\bcombat\b", r"\bengagement\b", r"\bexchang(e|ing)\s+fire",
        r"\bfight(ing|s)?\b.*\b(force|troops|militia|group)",
        r"\bgun\s*battle\b", r"\bambush",
    ],
    "sirens": [
        r"\bsiren[s]?\b", r"\brocket\s+alert", r"\bair[\-\s]?raid\s+alert",
        r"\balarm[s]?\b.*\b(sound|activ|triggered)",
        r"\bevacuat", r"\bshelter[s]?\b",
    ],
    "interception": [
        r"\bintercept(ed|ion|s)\b", r"\bshot\s+down\b",
        r"\bdown(ed|ing)\b.*\b(drone|missile|rocket|jet|aircraft)",
        r"\bengag(ed|ing)\b.*\b(drone|missile|target|with)",
        r"\bdefens(e|es)\s+(engag|intercept|activ)",
    ],
    "protest": [
        r"\bprotest", r"\bdemonstrat", r"\brally\b", r"\bunrest",
        r"\briot[s]?\b", r"\btear\s+gas",
    ],
    "ceasefire": [
        r"\bceasefire", r"\bcease\s*fire", r"\btruce\b", r"\barmistice",
        r"\bhostage\s+deal", r"\bprisoner\s+(swap|exchange)",
    ],
    "diplomatic": [
        r"\bsanction", r"\bnegotiat", r"\btreaty\b", r"\bsummit\b",
        r"\bdiplomat", r"\bUN\s+resolution", r"\bsecurity\s+council",
    ],
    "capture": [
        r"\bcaptur", r"\bseiz", r"\btaken\s+control", r"\bliberat",
        r"\brecaptur", r"\boverr(an|un)\b",
    ],
    "cyber": [
        r"\bcyber", r"\bhack(ed|ing)?\b", r"\bdata\s+breach", r"\bDDoS\b",
    ],
}

_compiled_events: dict[str, list[re.Pattern]] = {
    etype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for etype, patterns in EVENT_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Known locations -- common in OSINT/conflict reporting
# Matched as whole words, case-insensitive
# ---------------------------------------------------------------------------
KNOWN_LOCATIONS = [
    # Middle East
    "Gaza", "Rafah", "Khan Younis", "Khan Yunis", "Jabalia", "Beit Hanoun",
    "Deir al-Balah", "Nuseirat", "Tel Aviv", "Jerusalem", "Haifa",
    "West Bank", "Jenin", "Nablus", "Ramallah", "Hebron", "Tulkarm",
    "Beirut", "Dahiyeh", "Baalbek", "Tyre", "Sidon", "Nabatieh",
    "Damascus", "Aleppo", "Homs", "Idlib", "Latakia", "Deir ez-Zor",
    "Tehran", "Isfahan", "Tabriz", "Shiraz", "Baghdad", "Basra", "Erbil",
    "Mosul", "Kirkuk", "Sanaa", "Aden", "Hodeidah", "Marib",
    "Riyadh", "Jeddah", "Amman", "Cairo", "Sinai",
    # Ukraine / Eastern Europe
    "Kyiv", "Kharkiv", "Odessa", "Zaporizhzhia", "Kherson", "Mariupol",
    "Donetsk", "Luhansk", "Crimea", "Sevastopol", "Bakhmut", "Avdiivka",
    "Dnipro", "Lviv", "Sumy", "Kursk", "Belgorod", "Moscow",
    # Africa
    "Khartoum", "Mogadishu", "Tripoli", "Benghazi",
    # Strategic locations
    "Strait of Hormuz", "Red Sea", "Bab el-Mandeb", "Golan Heights",
    "Natanz", "Fordow", "Dimona", "Bushehr", "Parchin",
    "Al-Tanf", "Incirlik", "Al Udeid", "Ain al-Asad",
    "Suez Canal", "Persian Gulf", "Gulf of Oman",
    # General
    "Pentagon", "Kremlin", "White House", "United Nations",
]

_location_patterns = [
    re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE)
    for loc in KNOWN_LOCATIONS
]

# ---------------------------------------------------------------------------
# Geocoding lookup -- lat/lng for known locations
# ---------------------------------------------------------------------------
LOCATION_COORDS: dict[str, tuple[float, float]] = {
    # Middle East - Palestine/Israel
    "Gaza": (31.50, 34.47), "Rafah": (31.30, 34.25),
    "Khan Younis": (31.35, 34.30), "Khan Yunis": (31.35, 34.30),
    "Jabalia": (31.53, 34.48), "Beit Hanoun": (31.54, 34.53),
    "Deir al-Balah": (31.42, 34.35), "Nuseirat": (31.45, 34.39),
    "Tel Aviv": (32.08, 34.78), "Jerusalem": (31.77, 35.23),
    "Haifa": (32.82, 34.99), "West Bank": (31.95, 35.30),
    "Jenin": (32.46, 35.30), "Nablus": (32.22, 35.25),
    "Ramallah": (31.90, 35.20), "Hebron": (31.53, 35.10),
    "Tulkarm": (32.31, 35.03),
    # Lebanon
    "Beirut": (33.89, 35.50), "Dahiyeh": (33.85, 35.50),
    "Baalbek": (34.01, 36.21), "Tyre": (33.27, 35.20),
    "Sidon": (33.56, 35.37), "Nabatieh": (33.38, 35.48),
    # Syria
    "Damascus": (33.51, 36.29), "Aleppo": (36.20, 37.16),
    "Homs": (34.73, 36.71), "Idlib": (35.93, 36.63),
    "Latakia": (35.52, 35.78), "Deir ez-Zor": (35.34, 40.14),
    # Iran
    "Tehran": (35.69, 51.39), "Isfahan": (32.65, 51.68),
    "Tabriz": (38.08, 46.29), "Shiraz": (29.59, 52.58),
    # Iraq
    "Baghdad": (33.31, 44.37), "Basra": (30.51, 47.81),
    "Erbil": (36.19, 44.01), "Mosul": (36.34, 43.12),
    "Kirkuk": (35.47, 44.39),
    # Yemen
    "Sanaa": (15.37, 44.19), "Aden": (12.79, 45.02),
    "Hodeidah": (14.80, 42.95), "Marib": (15.46, 45.32),
    # Other Middle East
    "Riyadh": (24.71, 46.68), "Jeddah": (21.49, 39.19),
    "Amman": (31.95, 35.93), "Cairo": (30.04, 31.24),
    "Sinai": (29.50, 33.80),
    # Ukraine / Eastern Europe
    "Kyiv": (50.45, 30.52), "Kharkiv": (49.99, 36.23),
    "Odessa": (46.48, 30.73), "Zaporizhzhia": (47.84, 35.14),
    "Kherson": (46.64, 32.62), "Mariupol": (47.10, 37.55),
    "Donetsk": (48.00, 37.80), "Luhansk": (48.57, 39.33),
    "Crimea": (44.95, 34.10), "Sevastopol": (44.60, 33.52),
    "Bakhmut": (48.60, 38.00), "Avdiivka": (48.14, 37.74),
    "Dnipro": (48.46, 35.04), "Lviv": (49.84, 24.03),
    "Sumy": (50.91, 34.80), "Kursk": (51.73, 36.19),
    "Belgorod": (50.60, 36.59), "Moscow": (55.76, 37.62),
    # Africa
    "Khartoum": (15.50, 32.56), "Mogadishu": (2.05, 45.32),
    "Tripoli": (32.90, 13.18), "Benghazi": (32.12, 20.09),
    # Strategic locations
    "Strait of Hormuz": (26.56, 56.25), "Red Sea": (20.00, 38.00),
    "Bab el-Mandeb": (12.58, 43.33), "Golan Heights": (33.00, 35.80),
    "Natanz": (33.72, 51.73), "Fordow": (34.88, 51.26),
    "Dimona": (31.07, 35.15), "Bushehr": (28.97, 50.84),
    "Parchin": (35.52, 51.77),
    "Al-Tanf": (33.50, 38.67), "Incirlik": (37.00, 35.43),
    "Al Udeid": (25.12, 51.31), "Ain al-Asad": (33.80, 42.44),
    "Suez Canal": (30.46, 32.34), "Persian Gulf": (26.00, 52.00),
    "Gulf of Oman": (24.50, 58.50),
    # General / Buildings
    "Pentagon": (38.87, -77.06), "Kremlin": (55.75, 37.62),
    "White House": (38.90, -77.04), "United Nations": (40.75, -73.97),
}


def geocode_locations(location_names: list[str]) -> list[dict]:
    """Look up coordinates for a list of location names.
    Returns list of {name, lat, lng} for locations that have known coords."""
    results = []
    for name in location_names:
        coords = LOCATION_COORDS.get(name)
        if coords:
            results.append({"name": name, "lat": coords[0], "lng": coords[1]})
    return results

# ---------------------------------------------------------------------------
# Entity patterns -- organizations, military groups, leaders
# ---------------------------------------------------------------------------
KNOWN_ENTITIES = {
    # State military
    "IDF": "military", "IOF": "military",
    "IRGC": "military", "Quds Force": "military",
    "CENTCOM": "military", "AFRICOM": "military",
    "SDF": "military", "SAA": "military", "AFU": "military",
    "Wagner": "military", "PMC Wagner": "military",
    # Organizations
    "Hezbollah": "organization", "Hamas": "organization",
    "Houthis": "organization", "Ansar Allah": "organization",
    "Islamic Resistance": "organization", "Kata'ib Hezbollah": "organization",
    "Kataib Hezbollah": "organization", "PMF": "organization",
    "NATO": "organization", "UN": "organization", "UNRWA": "organization",
    "CIA": "organization", "Mossad": "organization", "FSB": "organization",
    "SBU": "organization", "GUR": "organization",
    "ISIS": "organization", "ISIL": "organization", "Daesh": "organization",
    "Al-Qaeda": "organization", "PKK": "organization", "YPG": "organization",
    # Weapons systems / platforms
    "Iron Dome": "system", "David's Sling": "system", "Arrow": "system",
    "Patriot": "system", "THAAD": "system", "S-300": "system", "S-400": "system",
    "Shahed": "system", "Shahab": "system",
    "B-52": "system", "F-35": "system", "F-16": "system",
    # Leaders
    "Netanyahu": "person", "Zelensky": "person", "Putin": "person",
    "Biden": "person", "Trump": "person", "Nasrallah": "person",
    "Sinwar": "person", "Khamenei": "person", "Raisi": "person",
    "Erdogan": "person", "Assad": "person", "Sisi": "person",
    "Macron": "person", "Scholz": "person", "Starmer": "person",
    "Gallant": "person", "Lavrov": "person", "Blinken": "person",
    "Sullivan": "person", "Austin": "person", "Ghalibaf": "person",
}

_entity_patterns = {
    name: label
    for name, label in KNOWN_ENTITIES.items()
}

# ---------------------------------------------------------------------------
# Sentiment keywords
# ---------------------------------------------------------------------------
_POS_WORDS = {
    "peace", "ceasefire", "agreement", "liberated", "rescued", "success",
    "victory", "humanitarian", "aid", "relief", "resolved", "stable",
    "truce", "deal", "released", "cooperation",
}
_NEG_WORDS = {
    "killed", "dead", "wounded", "destroyed", "attack", "explosion", "war",
    "crisis", "threat", "danger", "casualties", "bombing", "shelling",
    "missile", "conflict", "terror", "siege", "massacre", "devastat",
    "struck", "targeted", "violated", "escalation",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_event(text: str) -> str | None:
    """Return the most likely event type, or None if unclassified."""
    if not text:
        return None
    scores: dict[str, int] = {}
    for etype, patterns in _compiled_events.items():
        for pat in patterns:
            if pat.search(text):
                scores[etype] = scores.get(etype, 0) + 1
    if not scores:
        return None
    return max(scores, key=scores.get)


def extract_locations(text: str) -> list[str]:
    """Extract known location names from text."""
    if not text:
        return []
    found = []
    seen = set()
    for i, pat in enumerate(_location_patterns):
        if pat.search(text):
            loc = KNOWN_LOCATIONS[i]
            if loc.lower() not in seen:
                seen.add(loc.lower())
                found.append(loc)
    return found


def extract_entities(text: str) -> list[dict]:
    """Extract known entities (orgs, military, persons) from text."""
    if not text:
        return []
    found = []
    seen = set()
    for name, label in _entity_patterns.items():
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        if pattern.search(text) and name.lower() not in seen:
            seen.add(name.lower())
            found.append({"name": name, "label": label})
    return found


def score_sentiment(text: str) -> float:
    """Return a sentiment score from -1.0 (negative) to +1.0 (positive)."""
    if not text:
        return 0.0
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def extract_keywords(text: str) -> list[str]:
    """Pull out matching conflict/intel keywords found in the text."""
    if not text:
        return []
    found = set()
    for etype, patterns in _compiled_events.items():
        for pat in patterns:
            match = pat.search(text)
            if match:
                found.add(match.group(0).lower().strip())
    return sorted(found)


# ---------------------------------------------------------------------------
# Relevance scoring -- filter noise from actionable intelligence
# ---------------------------------------------------------------------------

# Strong indicators: concrete military/intel terms that alone indicate relevance
STRONG_INDICATORS = [
    # Named military/intel entities
    r"\bIRGC\b", r"\bQuds\s+Force\b", r"\bMossad\b", r"\bIDF\b", r"\bIOF\b",
    r"\bCENTCOM\b", r"\bHezbollah\b", r"\bHamas\b", r"\bHouthis?\b",
    r"\bAnsar\s+Allah\b", r"\bWagner\b", r"\bISIS\b", r"\bDaesh\b",
    r"\bKata.ib\s+Hezbollah\b", r"\bIslamic\s+Resistance\b",
    # Weapons systems
    r"\bIron\s+Dome\b", r"\bTHAAD\b", r"\bPatriot\b", r"\bShahed\b",
    r"\bS[\-\s]?[234]00\b", r"\bDavid.s\s+Sling\b", r"\bArrow\b",
    r"\bF[\-\s]?(?:16|35)\b", r"\bB[\-\s]?52\b",
    # Concrete action language
    r"\bstrike[s]?\s+(on|against|in|at|near)\b",
    r"\bmissile[s]?\s+(launch|attack|hit|struck|fired|intercepted)\b",
    r"\bdrone[s]?\s+(attack|strike|hit|struck|shot|downed|over)\b",
    r"\bair\s*strike\b", r"\bballistic\b", r"\bcruise\s+missile\b",
    r"\bintercept(ed|ion)\b", r"\bshelling\b", r"\bbombard(ed|ment)\b",
    r"\bground\s+(offensive|incursion|invasion|operation)\b",
    r"\bcasualt(y|ies)\b", r"\bdeath\s+toll\b",
    # Nuclear/strategic facilities
    r"\bnuclear\s+(facility|plant|program|enrichment|weapon)\b",
    r"\bNatanz\b", r"\bFordow\b", r"\bDimona\b", r"\bBushehr\b", r"\bParchin\b",
    # Strategic chokepoints
    r"\bStrait\s+of\s+Hormuz\b", r"\bBab\s+el[\-\s]Mandeb\b",
    r"\bSuez\s+Canal\b",
]

# Weak indicators: rhetoric/generic terms that only matter with context
WEAK_INDICATORS = [
    r"\bresistance\b", r"\bzionist[s]?\b", r"\bmartyr[s]?\b",
    r"\boccup(ation|ied|ier)\b", r"\baxis\s+of\s+resistance\b",
    r"\bimperial(ist|ism)\b", r"\baggress(ion|or)\b",
    r"\bvictory\b", r"\bdefeat\b", r"\bjihad\b",
    r"\bcoloni(al|alism|zer)\b", r"\boppression\b",
    r"\bGaza\b", r"\bPalestine\b", r"\bLiberation\b",
]

_strong_compiled = [re.compile(p, re.IGNORECASE) for p in STRONG_INDICATORS]
_weak_compiled = [re.compile(p, re.IGNORECASE) for p in WEAK_INDICATORS]


def score_relevance(text: str, entities: list[dict], event_type: str | None) -> float:
    """Score message relevance from 0.0 to 1.0.

    Rules:
    - Strong indicators alone make a message relevant (each adds 0.25)
    - Weak indicators add only 0.05 each
    - Having a classified event_type adds 0.20
    - Having recognized entities (military/org/system) adds 0.15
    - Messages need >= 0.30 to be considered relevant
    """
    if not text:
        return 0.0

    score = 0.0

    # Strong indicators
    for pat in _strong_compiled:
        if pat.search(text):
            score += 0.25

    # Weak indicators
    for pat in _weak_compiled:
        if pat.search(text):
            score += 0.05

    # Event type classified => concrete development
    if event_type:
        score += 0.20

    # Recognized entities present
    mil_or_org = sum(1 for e in entities if e.get("label") in ("military", "organization", "system"))
    score += min(mil_or_org * 0.10, 0.30)

    return min(round(score, 2), 1.0)


def parse_message(text: str) -> dict:
    """Run the full parsing pipeline on a message."""
    event_type = classify_event(text)
    entities = extract_entities(text)
    relevance = score_relevance(text, entities, event_type)
    return {
        "event_type": event_type,
        "sentiment": score_sentiment(text),
        "locations": extract_locations(text),
        "entities": entities,
        "keywords": extract_keywords(text),
        "relevance": relevance,
    }
