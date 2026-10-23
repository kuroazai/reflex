from __future__ import annotations

import numpy as np

from reflex import MacroEngine
from reflex.conditions import sprite_blob
from reflex.core import Region
from reflex.io import RecordingInputSink
from reflex.recipes import adjacent_sprite_face_talk, player_screen_xy_from_relative


class _SequencedFrames:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)

    def grab(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("frame queue exhausted")
        return self._frames.pop(0)


class _StepClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += float(dt)


def _frame_with_npc_at(npc_xy: tuple[int, int], *, w: int = 240, h: int = 160) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    nx, ny = npc_xy
    img[max(0, ny - 6) : ny + 6, max(0, nx - 6) : nx + 6, :] = 255
    return img


def test_adjacent_sprite_face_talk_west_neighbour_presses_left_then_a():
    player_xy = (120, 80)
    tile_size = 16
    npc_xy = (120 - tile_size, 80)

    frame_npc = _frame_with_npc_at(npc_xy)
    frame_npc2 = _frame_with_npc_at(npc_xy)
    frames = _SequencedFrames([frame_npc, frame_npc2])

    sprite_locator = sprite_blob(
        region=Region(0, 0, 240, 160),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
        min_area_px=20,
        max_area_px=10_000,
    )

    spec = adjacent_sprite_face_talk(
        name="west_npc",
        sprite_locator=sprite_locator,
        player_screen_xy=player_xy,
        tile_size_px=tile_size,
        talk_button="a",
        timeout_s=2.0,
        poll_hz=10.0,
        inter_press_ms=10,
    )

    sink = RecordingInputSink()
    clock = _StepClock()
    engine = MacroEngine(frames, sink, clock=clock.now, sleep=clock.sleep)
    outcome = engine.run(spec)

    assert outcome.status == "fired"
    assert outcome.last_action_seq == ("left", "a")
    assert sink.calls == [("left", 60), ("a", 60)]
    assert outcome.last_match_meta["tile_delta_cr"] == (-1, 0)


def test_adjacent_sprite_face_talk_far_npc_does_not_fire():
    player_xy = (120, 80)
    tile_size = 16
    npc_xy = (120 + 4 * tile_size, 80)

    frames = _SequencedFrames([_frame_with_npc_at(npc_xy) for _ in range(50)])

    sprite_locator = sprite_blob(
        region=Region(0, 0, 240, 160),
        hsv_lo=(0, 0, 200),
        hsv_hi=(179, 50, 255),
    )
    spec = adjacent_sprite_face_talk(
        name="far_npc",
        sprite_locator=sprite_locator,
        player_screen_xy=player_xy,
        tile_size_px=tile_size,
        talk_button="a",
        timeout_s=0.5,
        poll_hz=20.0,
    )
    sink = RecordingInputSink()
    clock = _StepClock()
    engine = MacroEngine(frames, sink, clock=clock.now, sleep=clock.sleep)
    outcome = engine.run(spec)

    assert outcome.status == "timed_out"
    assert sink.calls == []


def test_player_screen_xy_from_relative():
    px, py = player_screen_xy_from_relative(
        centroid_relative_xy=(0.5, 0.5),
        frame_size_wh=(1024, 736),
    )
    assert px == 512
    assert py == 368
