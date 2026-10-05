"""Versioned classifier prompt. Bump PROMPT_VERSION whenever the wording changes."""

import json

from models import RawItem
from timeutil import to_local

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """\
You classify local news items for a tracker of violent and property crime in the
City of St. Louis, Missouri. The input is a JSON array of news items, each with a
raw_item_id. Reply with a JSON array containing exactly one result per input item,
matching the given schema. Judge each item on its own.

Fields of each result:
- raw_item_id: copied unchanged from the input item.
- is_crime: true only if the item reports a specific, recent shooting, burglary, or
  homicide (killing) that happened within the past few days.
- crime_type: "shooting", "burglary", or "homicide"; null when is_crime is false.
  If someone was shot and killed, use "homicide".
- in_stl: true only if the crime happened inside the City of St. Louis (the
  independent city). St. Louis County and its municipalities (e.g. Ferguson,
  Florissant, Jennings, Clayton, University City, Normandy, Berkeley), St. Charles,
  Jefferson County, and the Illinois Metro East (e.g. East St. Louis, Belleville)
  are NOT in the city: in_stl=false. If the location is unclear, in_stl=false.
- occurred_at: when the crime happened (ISO 8601 with UTC offset, St. Louis local
  time), resolved against the publish date (e.g. "Saturday night" -> that Saturday
  around 21:00). Never default to the publish date. null if unknown.
- location: the most specific place mentioned (address, intersection, or
  neighborhood), or null.
- confidence: 0 to 1, how sure you are that is_crime, crime_type, and in_stl are right.

Set is_crime=false for:
- Court news: trials, verdicts, sentencing, pleas, appeals, hearings.
- Arrests, charges, or identifications for crimes that happened weeks or more ago.
- Crimes outside the City of St. Louis (also set in_stl=false).
- Opinion pieces, editorials, columns, and letters.
- Crime statistics, trend reports, policy, and politics stories.
- Anything that is not a shooting, burglary, or homicide (sports, weather,
  accidents, fires, overdoses, natural deaths).
"""


def build_user_prompt(items: list[RawItem]) -> str:
    """JSON array of the items to classify; results are matched back by raw_item_id."""
    return json.dumps(
        [
            {
                "raw_item_id": item.id,
                # St. Louis time, so "Friday night" resolves to the right local day.
                "published_at": (
                    to_local(item.published_at).isoformat() if item.published_at else None
                ),
                "title": item.title,
                "body": item.body or "",
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=1,
    )
