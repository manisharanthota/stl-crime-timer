"""Official City of St. Louis neighborhoods, and normalizing free-text names onto them.

Source: https://www.stlouis-mo.gov/live-work/community/neighborhoods/index.cfm
(retrieved 2026-10-05; the page's " Neighborhood" suffixes are dropped).
"""

import re

NEIGHBORHOODS = (
    "Academy",
    "Baden",
    "Benton Park",
    "Benton Park West",
    "Bevo Mill",
    "Botanical Heights",
    "Boulevard Heights",
    "Carondelet",
    "Carr Square",
    "Central West End",
    "Cheltenham",
    "Clayton-Tamm",
    "Clifton Heights",
    "College Hill",
    "Columbus Square",
    "Compton Heights",
    "Covenant Blu Grand Center",
    "DeBaliviere Place",
    "Downtown",
    "Downtown West",
    "Dutchtown",
    "Ellendale",
    "Fairground",
    "Forest Park Southeast",
    "Fountain Park",
    "Fox Park",
    "Franz Park",
    "Gravois Park",
    "Hamilton Heights",
    "Hi Pointe",
    "Holly Hills",
    "Hyde Park",
    "Jeff-Vander-Lou",
    "Kings Oak",
    "Kingsway East",
    "Kingsway West",
    "Kosciusko",
    "Lafayette Square",
    "LaSalle Park",
    "Lewis Place",
    "Lindenwood Park",
    "Marine Villa",
    "Mark Twain",
    "Mark Twain I-70 Industrial",
    "McKinley Heights",
    "Midtown",
    "Mount Pleasant",
    "Near North Riverfront",
    "North Hampton",
    "North Pointe",
    "North Riverfront",
    "O'Fallon",
    "Old North St. Louis",
    "Patch",
    "Peabody Darst Webbe",
    "Penrose",
    "Princeton Heights",
    "Riverview",
    "Shaw",
    "Skinker DeBaliviere",
    "Soulard",
    "Southampton",
    "Southwest Garden",
    "St. Louis Hills",
    "St. Louis Place",
    "The Gate District",
    "The Greater Ville",
    "The Hill",
    "The Ville",
    "Tiffany",
    "Tower Grove East",
    "Tower Grove South",
    "Vandeventer",
    "Visitation Park",
    "Walnut Park East",
    "Walnut Park West",
    "Wells Goodfellow",
    "West End",
    "Wydown Skinker",
)

# Common informal names -> official name.
_ALIASES = {
    "cwe": "Central West End",
    "grand center": "Covenant Blu Grand Center",
    "jvl": "Jeff-Vander-Lou",
    "old north": "Old North St. Louis",
    "mcree town": "Botanical Heights",
    "downtown st louis": "Downtown",
}


def _key(name: str) -> str:
    """Lowercase, punctuation to spaces, "saint" -> "st", no leading "the" or
    trailing "neighborhood"."""
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"^the ", "", text)
    text = re.sub(r" neighborhood$", "", text)
    return text


_BY_KEY = {_key(n): n for n in NEIGHBORHOODS}
_BY_KEY.update({_key(alias): name for alias, name in _ALIASES.items()})


def normalize_neighborhood(name: str | None) -> str | None:
    """The official neighborhood name for `name`, or None if it isn't one."""
    if not name:
        return None
    return _BY_KEY.get(_key(name))
