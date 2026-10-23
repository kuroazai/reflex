"""
reflex — typed reflex runtime for fast event-driven game / GUI control.

Two-tier control:

  slow tier (your agent / planner)   fast tier (this package)
  -----------------------------      -------------------------------------
  scene reasoning, planning,    →    event-driven reflexes with priorities,
  macro compilation                  gates, guards, verification, deadman
  (LLM in the loop)                  (LLM never on the hot path)

Project-agnostic: the engine knows nothing about the program it drives.
Screen capture and input are pluggable; Windows backends live in
`reflex.backends`.

Public surface:

    reflex.core       FrameView, Match, Region, Predicate, TriggerSpec, TriggerOutcome
    reflex.actions    Key, KeyDown, KeyUp, Chord, Click, Wait + factory helpers
    reflex.conditions sprite_blob, template_match, color_at, pixel_at, tile_occupied,
                      motion_present, region_color_ratio, extras_equals, any_of/all_of/not_ …
    reflex.expect     gone, appears, changed — postconditions that verify a fire
    reflex.recipes    on_match, click_target, dialog_advance_loop, adjacent_*_face_talk …
    reflex.watchdog   anti_stuck, deadman_release
    reflex.runtime    ReflexRuntime — schedule, gate, guard, fire, verify, report
    reflex.engine     MacroEngine — run one trigger → TriggerOutcome
    reflex.registry   plug-in registries (predicate / recipe / expect / sink / source)
    reflex.dsl        YAML → TriggerSpecs via the registry
    reflex.replay     scenario-based validation over recorded frames
    reflex.stats      summarise an events.jsonl stream
"""

from reflex.actions import Chord, Click, Key, KeyDown, KeyUp, Wait  # noqa: F401
from reflex.conditions import (  # noqa: F401
    all_,
    any_,
    not_,
    pixel_at,
    region_color_ratio,
    sprite_blob,
    template_match,
    tile_occupied,
)
from reflex.core import FrameView, Match, Predicate, Region, TriggerOutcome, TriggerSpec  # noqa: F401
from reflex.engine import MacroEngine  # noqa: F401
from reflex.io import FrameSource, InputSink, LogCallback  # noqa: F401
from reflex.runtime import ReflexRuntime  # noqa: F401

__all__ = [
    "Chord",
    "Click",
    "FrameSource",
    "FrameView",
    "InputSink",
    "Key",
    "KeyDown",
    "KeyUp",
    "LogCallback",
    "MacroEngine",
    "Match",
    "Predicate",
    "ReflexRuntime",
    "Region",
    "TriggerOutcome",
    "TriggerSpec",
    "Wait",
    "all_",
    "any_",
    "not_",
    "pixel_at",
    "region_color_ratio",
    "sprite_blob",
    "template_match",
    "tile_occupied",
]

__version__ = "0.2.0"
