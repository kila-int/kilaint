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
        r"\bprotest", r"\bdemonstrat", r"\brally\b", r"\bralli(ed|es)\b",
        r"\bunrest", r"\briot[s]?\b", r"\btear\s+gas",
        r"\bmarching\b", r"\bmarch(ed|es)?\s+(against|on|to|in|through)\b",
        r"\bchant(ed|ing|s)?\b", r"\bslogans?\b",
        r"\btook\s+to\s+the\s+streets\b", r"\bstreet\s+protest",
        r"\banti[\-\s]?(war|government|american|israel|US|western)\b",
        r"\bpro[\-\s]?(palestine|iran|resistance)\b",
        r"\bpublic\s+(outrage|anger|fury|outcry)\b",
        r"\bsit[\-\s]?in\b", r"\bstrike\s+action\b", r"\bgeneral\s+strike\b",
        r"\bsolidarity\s+(march|rally|protest|gathering)\b",
        r"\bmass\s+(gathering|rally|protest|demonstration)\b",
        r"\bflag[\-\s]?burning\b", r"\beffig(y|ies)\b",
    ],
    "political": [
        r"\belect(ed|ion|ions|oral)\b", r"\bvot(e|ed|ing|ers?)\b",
        r"\bparliament", r"\bcongress\b", r"\blegislat",
        r"\bprime\s+minister\b", r"\bpresident\s+(elect|appoint|announc|sworn|inaugurat)",
        r"\bsupreme\s+leader\b", r"\bayatollah\b",
        r"\bcabinet\b", r"\bminister\s+of\b", r"\bforeign\s+minister\b",
        r"\bdefense\s+minister\b", r"\bgovernment\s+(form|collaps|resign|fell|announc)",
        r"\bcoup\b", r"\bregime\s+change\b", r"\bpower\s+transfer\b",
        r"\bsworn\s+in\b", r"\binaugurat", r"\bimpeach",
        r"\bpolitical\s+(crisis|turmoil|shakeup|shift|transition)",
        r"\bresign(ed|ation|s)?\b", r"\bappoint(ed|ment|s)?\b",
        r"\bsuccessor\b", r"\bsuccession\b",
        r"\bveto(ed)?\b", r"\bexecutive\s+order\b",
        r"\bparty\b.*\b(leader|win|lost|coalition)\b",
    ],
    "humanitarian": [
        r"\bhumanitarian\s+(crisis|aid|corridor|disaster|catastroph)",
        r"\brefugee[s]?\b", r"\bdisplac(ed|ement)\b", r"\bIDP[s]?\b",
        r"\bfamine\b", r"\bstarvation\b", r"\bfood\s+(shortage|crisis|insecurity)",
        r"\baid\s+(convoy|shipment|deliver|block|denied|cut)",
        r"\bblockade\b", r"\bsiege\b", r"\bembargo\b",
        r"\bwar\s+crime[s]?\b", r"\bgenocid", r"\bethnic\s+cleansing",
        r"\bcivilian\s+(casualties|deaths|killed|targeted|suffering)",
        r"\bhospital\s+(struck|hit|bombed|destroyed|attacked)",
        r"\bschool\s+(struck|hit|bombed|destroyed|attacked)",
        r"\bUN\s+(aid|relief|agency|warning|report)",
        r"\bRed\s+Cross\b", r"\bRed\s+Crescent\b", r"\bMSF\b",
        r"\bforced\s+(displacement|evacuation|migration)\b",
    ],
    "sanctions_economic": [
        r"\bsanction(s|ed|ing)?\b", r"\btrade\s+(ban|war|restriction|embargo)",
        r"\basset\s+(freez|seiz)", r"\bblacklist",
        r"\barms\s+(embargo|deal|sale|transfer|shipment)",
        r"\boil\s+(embargo|sanction|price|export|import)",
        r"\beconomic\s+(war|pressure|colaps|crisis|isolation)",
        r"\bban(ned|s)?\s+(import|export|trade|travel)\b",
        r"\bfreez(e|ing)\s+(assets|funds|accounts)\b",
        r"\bfinancial\s+(sanction|restriction|penalty)\b",
    ],
    "ceasefire": [
        r"\bceasefire", r"\bcease\s*fire", r"\btruce\b", r"\barmistice",
        r"\bhostage\s+deal", r"\bprisoner\s+(swap|exchange)",
        r"\bpeace\s+(talk|deal|agreement|process|plan|proposal|negotiation)",
        r"\bde[\-\s]?escalat", r"\bcalm\s+restored",
    ],
    "diplomatic": [
        r"\bsanction", r"\bnegotiat", r"\btreaty\b", r"\bsummit\b",
        r"\bdiplomat", r"\bUN\s+resolution", r"\bsecurity\s+council",
        r"\bambassador\b", r"\bembass(y|ies)\b", r"\bconsulate\b",
        r"\bbilateral\b", r"\bmultilateral\b",
        r"\bforeign\s+(policy|affairs|relations|ministry)\b",
        r"\bstate\s+visit\b", r"\bphone\s+call\b.*\b(leader|president|minister)",
        r"\bjoint\s+statement\b", r"\bcommun(ique|iqué)\b",
        r"\balliance\b", r"\bpact\b", r"\bagreement\s+(sign|reach|broker)",
        r"\bsever(ed)?\s+(ties|relations|diplomatic)\b",
        r"\brecall(ed)?\s+ambassador\b", r"\bexpel(led)?\s+diplomat\b",
        r"\bICJ\b", r"\bICC\b", r"\bGeneva\s+Convention\b",
    ],
    "intelligence": [
        r"\bintelligence\s+(report|brief|gather|shar|failure|warn|assess)",
        r"\bespionage\b", r"\bspy\b", r"\bcovert\s+(op|operation|mission|action)",
        r"\bassassinat", r"\btarget(ed)?\s+killing",
        r"\bsurveillance\b", r"\binterrog", r"\bdetain(ed|ee|tion)?\b",
        r"\barrest(ed|s)?\b.*\b(spy|agent|operative|suspect|cell)\b",
        r"\bsleeper\s+cell\b", r"\bsabotag",
        r"\bwhistleblow", r"\bleak(ed|s)?\s+(document|intel|classified)",
    ],
    "nuclear": [
        r"\bnuclear\b", r"\buranium\s+(enrich|stockpile|centrifuge)",
        r"\bcentrifuge[s]?\b", r"\bplutonium\b",
        r"\bIAEA\b", r"\bnon[\-\s]?proliferation\b", r"\bNPT\b",
        r"\bnuclear\s+(deal|agreement|talks|program|weapon|warhead|test|threat|facility|plant|site|inspect)",
        r"\bJCPOA\b", r"\batomic\b",
        r"\bbreakout\s+(time|capability)\b",
    ],
    "capture": [
        r"\bcaptur", r"\bseiz", r"\btaken\s+control", r"\bliberat",
        r"\brecaptur", r"\boverr(an|un)\b",
    ],
    "cyber": [
        r"\bcyber", r"\bhack(ed|ing)?\b", r"\bdata\s+breach", r"\bDDoS\b",
    ],
    "threat": [
        r"\bthreat(en|ened|ens|s)?\s+(to|of|against)\b",
        r"\bwarn(ed|ing|s)?\s+(of|against|about|that)\b",
        r"\bultimatum\b", r"\bred\s+line\b",
        r"\bescalat(e|ed|ion|ing)\b", r"\btension[s]?\b",
        r"\bprovoc(ation|ative)\b", r"\bsabre[\-\s]?rattling\b",
        r"\bon\s+(high\s+)?alert\b", r"\bready\s+to\s+(strike|respond|retaliat)",
        r"\bretaliat", r"\brevenge\b", r"\baveng",
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
    # Iran - more cities/ports
    "Bandar Abbas", "Bander Abbas", "Bandar Lengeh", "Bander Lengeh",
    "Chabahar", "Ahvaz", "Kerman", "Mashhad", "Qom", "Arak",
    "Sistan", "Baluchistan", "Khuzestan", "Hormozgan",
    "Bandar Imam Khomeini", "Abadan", "Khorramshahr",
    # Turkey / Cyprus
    "Adana", "Ankara", "Istanbul", "Diyarbakir", "Hatay",
    "Mersin", "Gaziantep", "Izmir", "Antalya",
    "Cyprus", "Nicosia", "Larnaca",
    # More Lebanon
    "Tripoli Lebanon", "Jounieh", "Bekaa", "Hermel",
    "Khiam", "Marjayoun", "Bint Jbeil",
    # More Palestine/Israel
    "Ashkelon", "Ashdod", "Beer Sheva", "Beersheba",
    "Eilat", "Netanya", "Nazareth", "Tiberias",
    "Sderot", "Kiryat Shmona", "Metula", "Acre",
    "Khan al-Ahmar", "Bethlehem", "Jericho", "Qalqilya",
    "Beit Lahia", "Shifa", "Al-Shifa",
    # More Iraq
    "Fallujah", "Ramadi", "Tikrit", "Samarra", "Najaf", "Karbala",
    "Taji", "Balad", "Al-Asad", "Camp Buehring",
    # More Syria
    "Raqqa", "Palmyra", "Daraa", "Tartus", "Qamishli", "Manbij",
    "Abu Kamal", "Deir Ezzor",
    # Jordan
    "Zarqa", "Irbid", "Aqaba", "Mafraq",
    # Saudi Arabia
    "Dammam", "Dhahran", "Tabuk", "Medina", "Yanbu", "Jubail",
    # More Yemen
    "Taiz", "Saada", "Dhamar", "Al Bayda", "Hajjah",
    # Strategic locations
    "Strait of Hormuz", "Red Sea", "Bab el-Mandeb", "Golan Heights",
    "Natanz", "Fordow", "Dimona", "Bushehr", "Parchin",
    "Al-Tanf", "Incirlik", "Al Udeid", "Ain al-Asad",
    "Suez Canal", "Persian Gulf", "Gulf of Oman",
    "Arabian Sea", "Mediterranean",
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
    # Iran - more cities/ports
    "Bandar Abbas": (27.19, 56.27), "Bander Abbas": (27.19, 56.27),
    "Bandar Lengeh": (26.56, 54.88), "Bander Lengeh": (26.56, 54.88),
    "Chabahar": (25.29, 60.64), "Ahvaz": (31.32, 48.68),
    "Kerman": (30.28, 57.08), "Mashhad": (36.30, 59.60),
    "Qom": (34.64, 50.88), "Arak": (34.09, 49.69),
    "Sistan": (31.00, 61.50), "Baluchistan": (27.00, 63.00),
    "Khuzestan": (31.50, 49.00), "Hormozgan": (27.00, 56.00),
    "Bandar Imam Khomeini": (30.43, 49.08), "Abadan": (30.34, 48.30),
    "Khorramshahr": (30.44, 48.17),
    # Turkey / Cyprus
    "Adana": (37.00, 35.32), "Ankara": (39.93, 32.86),
    "Istanbul": (41.01, 28.98), "Diyarbakir": (37.91, 40.24),
    "Hatay": (36.40, 36.35), "Mersin": (36.80, 34.63),
    "Gaziantep": (37.07, 37.38), "Izmir": (38.42, 27.14),
    "Antalya": (36.90, 30.70),
    "Cyprus": (35.13, 33.43), "Nicosia": (35.17, 33.36),
    "Larnaca": (34.92, 33.63),
    # More Lebanon
    "Tripoli Lebanon": (34.44, 35.83), "Jounieh": (33.98, 35.62),
    "Bekaa": (33.85, 36.00), "Hermel": (34.39, 36.39),
    "Khiam": (33.36, 35.64), "Marjayoun": (33.36, 35.59),
    "Bint Jbeil": (33.12, 35.43),
    # More Palestine/Israel
    "Ashkelon": (31.67, 34.57), "Ashdod": (31.80, 34.65),
    "Beer Sheva": (31.25, 34.79), "Beersheba": (31.25, 34.79),
    "Eilat": (29.56, 34.95), "Netanya": (32.33, 34.86),
    "Nazareth": (32.70, 35.30), "Tiberias": (32.79, 35.53),
    "Sderot": (31.52, 34.60), "Kiryat Shmona": (33.21, 35.57),
    "Metula": (33.28, 35.58), "Acre": (32.93, 35.08),
    "Khan al-Ahmar": (31.80, 35.33), "Bethlehem": (31.70, 35.21),
    "Jericho": (31.86, 35.46), "Qalqilya": (32.19, 34.97),
    "Beit Lahia": (31.55, 34.50), "Shifa": (31.52, 34.45),
    "Al-Shifa": (31.52, 34.45),
    # More Iraq
    "Fallujah": (33.35, 43.78), "Ramadi": (33.43, 43.31),
    "Tikrit": (34.61, 43.68), "Samarra": (34.20, 43.87),
    "Najaf": (32.00, 44.34), "Karbala": (32.62, 44.02),
    "Taji": (33.52, 44.26), "Balad": (34.01, 44.15),
    "Al-Asad": (33.78, 42.44), "Camp Buehring": (29.33, 47.66),
    # More Syria
    "Raqqa": (35.95, 39.01), "Palmyra": (34.56, 38.27),
    "Daraa": (32.63, 36.10), "Tartus": (34.89, 35.89),
    "Qamishli": (37.05, 41.23), "Manbij": (36.53, 37.96),
    "Abu Kamal": (34.45, 40.92), "Deir Ezzor": (35.34, 40.14),
    # Jordan
    "Zarqa": (32.07, 36.09), "Irbid": (32.56, 35.85),
    "Aqaba": (29.53, 35.01), "Mafraq": (32.34, 36.21),
    # Saudi Arabia
    "Dammam": (26.43, 50.10), "Dhahran": (26.30, 50.14),
    "Tabuk": (28.38, 36.57), "Medina": (24.47, 39.61),
    "Yanbu": (24.09, 38.06), "Jubail": (27.01, 49.66),
    # More Yemen
    "Taiz": (13.58, 44.02), "Saada": (16.94, 43.76),
    "Dhamar": (14.54, 44.40), "Al Bayda": (14.17, 45.57),
    "Hajjah": (15.69, 43.60),
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
    "Arabian Sea": (15.00, 65.00), "Mediterranean": (35.00, 18.00),
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
    # Political/leadership developments
    r"\bsupreme\s+leader\b", r"\belect(ed|ion)\b.*\b(president|leader|minister)",
    r"\bcoup\b", r"\bregime\s+change\b", r"\bassassinat",
    r"\bwar\s+crime\b", r"\bgenocid", r"\bICC\b", r"\bICJ\b",
    r"\bJCPOA\b", r"\bIAEA\b", r"\buranium\s+enrich",
    # Escalation language
    r"\bescalat(ion|ed|ing)\b", r"\bretaliat", r"\bultimatum\b",
    r"\bdeclar(e|ed|ation)\s+of\s+war\b",
    # Humanitarian crises
    r"\bblockade\b", r"\bfamine\b", r"\bhumanitarian\s+crisis\b",
    r"\brefugee\s+(crisis|flood|wave)\b",
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
