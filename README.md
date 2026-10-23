# reflex

A standalone, project-agnostic **frame-trigger macro engine** for any program that needs to react to a live image stream faster than an LLM-in-the-loop can.

The model:

- **Frame source** (a callable that returns a BGR `numpy` array) and **input sink** (a callable that fires a button press) are pluggable. `reflex` does not know whether it is driving an emulator, a game, a website, or a robot.
- A **Trigger** is a `(predicate, action_factory)` pair that the engine polls against each captured frame at a configurable rate (default 30 Hz).
- The **Engine** owns the poll loop. When a predicate matches, it re-verifies on a fresh frame, then fires the action sequence with sub-100 ms latency. After `max_fires` (default 1) it stops and returns a structured `TriggerOutcome`.
- Every armed trigger has a **timeout** (deadman). The engine cannot run indefinitely.
- Predicates and actions compose. `all_`, `any_`, `not_` for boolean logic; `ComputeThenPress` for actions whose button sequence depends on the matched location.

This is intentionally a small primitive. It is **not** a workflow framework, **not** an event bus, **not** a planner. The intended pattern is:

```
LLM / planner   →   defines TriggerSpec   →   reflex engine watches & fires   →   structured outcome   →   LLM resumes
```

## Install (editable, local consumer)

```bash
pip install -e L:\reflex
```

## Minimal example

```python
import time
import numpy as np
from reflex import MacroEngine, TriggerSpec, Match
from reflex.conditions import pixel_at

def grab():
    return np.zeros((160, 240, 3), dtype=np.uint8)

def press(button: str, *, hold_ms: int) -> bool:
    print(f"PRESS {button} for {hold_ms}ms")
    return True

def predicate(view):
    rgb_match = pixel_at(x=120, y=80, rgb=(0, 0, 0), tolerance=10)
    return Match() if rgb_match(view) else None

spec = TriggerSpec(
    name="example",
    predicate=predicate,
    action_factory=lambda match: ["a"],
    timeout_s=2.0,
)

engine = MacroEngine(frame_source=grab, input_sink=press)
outcome = engine.run(spec)
print(outcome)
```

## Why this exists

LLM agents reasoning at 5–15 s per decision cannot interact with wandering NPCs, dialogue boxes, or rapid UI events. `reflex` lets the LLM **hand off** specific reactive tasks to a pure-Python loop that runs at ~30 Hz, then resume planning when the trigger fires (or expires).

## Status

`v0.1` — core engine, primitive predicates, intercept recipe, NPC face-talk consumer in [`aerarium`](file:///L:/aerarium).

Planned: OCR predicate, watcher / background loops, declarative YAML triggers, recording / replay, priority queue across multiple armed triggers.
