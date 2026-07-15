"""Unified class map for the v1 coarse taxonomy (project_plan.md §2).

Every source-dataset converter imports UNIFIED_CLASSES and its own
`*_CLASS_MAP` dict from here rather than hardcoding ids, so the whole
project agrees on one class list and one remap table.

A source class id/name that maps to None is intentionally DROPPED
(not an object we detect, e.g. sports fields, harbors, runways).
A source class id/name that is simply ABSENT from the map is treated
as unverified -> the converters raise instead of silently mis-mapping.
"""

UNIFIED_CLASSES = {
    0: "person",
    1: "vehicle",
    2: "military_equipment",
}
NAME_TO_ID = {name: idx for idx, name in UNIFIED_CLASSES.items()}

DROP = None

# --- VEDAI -----------------------------------------------------------
# Annotation lines carry a numeric class id (no name string). Two
# independent third-party reproductions of the devkit disagreed on some
# ids, resolved against the actual per-image counts in this download
# (ids 3/6/12 never occur at all; id 11 has 955 instances, too common
# to be a rare/unused duplicate, confirming it's "pickup" not "car dup"):
#   1 car, 2 truck, 4 tractor, 5 camping car, 7 motorcycle, 8 bus,
#   9 van, 10 other, 11 pickup, 23 boat, 31 plane
# boat (23) has no natural bucket in the coarse v1 taxonomy; mapped to
# vehicle
VEDAI_CLASS_MAP = {
    1: "vehicle",   # car
    2: "vehicle",   # truck
    4: "vehicle",   # tractor
    5: "vehicle",   # camping car
    7: "vehicle",   # motorcycle
    8: "vehicle",   # bus
    9: "vehicle",   # van
    10: "vehicle",  # other
    11: "vehicle",  # pickup
    23: "vehicle",  # boat
    31: "military_equipment",  # plane (no civilian-aircraft bucket in v1)
}

# --- KIIT-MiTA ---------------------------------------------------------
# Already shipped as YOLO txt with its own 7-class KIIT-MiTA.yml; this
# just remaps those class ids into the unified 3-class taxonomy.
KIIT_CLASS_MAP = {
    0: "military_equipment",  # Artilary [sic]
    1: "military_equipment",  # Missile
    2: "military_equipment",  # Radar
    3: "military_equipment",  # M. Rocket Launcher
    4: "person",               # Soldier
    5: "military_equipment",  # Tank
    6: "vehicle",              # Vehicle
}

# --- VisDrone2019-DET --------------------------------------------------
# Standard VisDrone category ids. category 0 ("ignored regions", almost
# always paired with score=0) is dropped -- it marks areas to exclude
# from evaluation, not an object. category 11 ("others") is an
# unspecified catch-all with no clear unified bucket, dropped rather
# than guessed. bicycle/tricycle/awning-tricycle (3/7/8) dropped
VISDRONE_CLASS_MAP = {
    0: DROP,                 # ignored regions
    1: "person",             # pedestrian
    2: "person",             # people
    3: DROP,                 # bicycle
    4: "vehicle",            # car
    5: "vehicle",            # van
    6: "vehicle",            # truck
    7: DROP,                 # tricycle
    8: DROP,                 # awning-tricycle
    9: "vehicle",            # bus
    10: "vehicle",           # motor
    11: DROP,                # others
}
