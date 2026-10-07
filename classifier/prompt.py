"""Versioned classifier prompt. Bump PROMPT_VERSION whenever the wording changes."""

import json

from matcher.neighborhoods import NEIGHBORHOODS
from models import RawItem
from timeutil import to_local

PROMPT_VERSION = "v7"

SYSTEM_PROMPT = """\
You classify local news items for a tracker of violent and property crime in the
City of St. Louis, Missouri. The input is a JSON array of news items, each with a
raw_item_id. Reply with a JSON array containing exactly one result per input item,
matching the given schema. Judge each item on its own.

Fields of each result:
- raw_item_id: copied unchanged from the input item.
- is_crime: true only if the item is about a specific shooting, burglary, or
  homicide (killing) that happened within 7 days before the publish date. That
  includes follow-up stories about such a recent crime: the victim identified,
  family or friends remember the victim, a vigil, police seek or arrest a suspect.
- is_followup: true if the item is a follow-up story like those (published after
  the crime and mainly about its aftermath) rather than the first report of it.
  An item mainly about an arrest, a detention, a suspect taken into custody, or a
  "person of interest" is always a follow-up (is_followup=true), even if it is the
  first story you see about that crime: an arrest is not a new crime.
  false when is_crime is false.
- crime_type: "shooting", "burglary", or "homicide"; null when is_crime is false.
  If someone was shot and killed, use "homicide".
- was_shooting: true if anyone was shot with a gun, including a fatal shooting
  (crime_type "homicide", was_shooting true). false for killings by other means
  (stabbing, beating, strangling), for burglaries, and when is_crime is false.
- in_stl: true only if the crime happened inside the City of St. Louis (the
  independent city). St. Louis County and its municipalities (e.g. Ferguson,
  Florissant, Jennings, Clayton, University City, Normandy, Berkeley), St. Charles,
  Jefferson County, and the Illinois Metro East (e.g. East St. Louis, Belleville)
  are NOT in the city: in_stl=false. If the location is unclear, in_stl=false.
- occurred_at: when the crime happened (ISO 8601 with UTC offset, St. Louis local
  time), resolved against the publish date (e.g. "Saturday night" -> that Saturday
  around 21:00). For follow-up stories, this is when the original crime happened,
  never when the arrest, detention, or identification happened and never when the
  story was published. If a follow-up doesn't say when the crime itself happened,
  occurred_at is null. Never default to the publish date. null if unknown.
  For a string or series of crimes reported together (e.g. "four burglaries
  Wednesday through Sunday"), occurred_at is the time of the latest one.
  occurred_at is never later than the item's published_at: a crime can't happen
  after the story about it was published.
- time_precision: how much of occurred_at the item states.
  "exact": it gives a clock time ("12:40 a.m.", "around 9:30 p.m.").
  "date_only": it gives only the day, or a part of the day ("Friday",
  "early Friday", "Saturday night"); occurred_at is your estimate. If you put
  00:00 because no time was given, this is date_only.
  "unknown": no date at all; occurred_at is null.
- location: the most specific place mentioned (address, intersection, or
  neighborhood), or null.
- neighborhood: the official City of St. Louis neighborhood where the crime
  happened, spelled exactly as in the list below, or null if the item doesn't say
  or you aren't sure. Use the neighborhood the item names, or the one containing
  the address it gives.
- confidence: 0 to 1, how sure you are that is_crime, crime_type, and in_stl are right.

Set is_crime=false for:
- Court news, even about a recent crime: charges filed, trials, verdicts,
  sentencing, pleas, appeals, hearings.
- Follow-ups (identifications, memorials, vigils, arrests, searches for suspects)
  about crimes that happened more than 7 days before the publish date.
- Crimes outside the City of St. Louis (also set in_stl=false).
- Opinion pieces, editorials, columns, and letters.
- Crime statistics, trend reports, policy, and politics stories.
- Anything that is not a shooting, burglary, or homicide (sports, weather,
  accidents, fires, overdoses, natural deaths).

Official City of St. Louis neighborhoods (for the neighborhood field):
""" + "\n".join(f"- {name}" for name in NEIGHBORHOODS) + "\n"


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
