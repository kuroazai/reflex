"""Windows capture and input against a real window this test creates.

Nothing here touches the user's own windows, moves the real cursor or takes
focus. The SendInput round trip does all three, so it only runs with
REFLEX_LIVE_INPUT=1.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows backend")

if sys.platform == "win32":
    from reflex.backends.windows import (
        CaptureError,
        PostMessageSink,
        ScreenSource,
        SendInputSink,
        UnknownKey,
        WindowNotFound,
        WindowSource,
        find_window,
        list_windows,
        screen_capture_available,
    )
    from reflex.backends.windows.keys import key_lparam, resolve, vk_for
    from reflex.backends.windows.testwindow import BLUE_BGR, RED_BGR, TestWindow


def _needs_screen():
    if not screen_capture_available():
        pytest.skip("screen is locked: desktop copies are refused until it is unlocked")


@pytest.fixture
def tw():
    with TestWindow(title=f"reflex-test-{os.getpid()}-{time.monotonic_ns()}") as win:
        yield win


def _near(px, bgr, tol=6):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(px, bgr, strict=True))


# --------------------------------------------------------------------------- keys


def test_key_names_resolve_case_insensitively():
    assert vk_for("A") == vk_for("a") == 0x41
    assert vk_for("Enter") == vk_for("return") == 0x0D
    assert vk_for("f12") == 0x7B
    assert vk_for("numpad5") == 0x65
    assert resolve("up").extended and not resolve("w").extended


def test_unknown_key_names_the_problem():
    with pytest.raises(UnknownKey, match="unknown key 'notakey'"):
        vk_for("notakey")


def test_key_lparam_bits():
    down = key_lparam(0x1E, extended=False, up=False)
    up = key_lparam(0x1E, extended=True, up=True)
    assert down == 1 | (0x1E << 16)
    assert up & (1 << 24) and up & (1 << 30) and up & (1 << 31)


# --------------------------------------------------------------------------- windows


def test_find_window_by_title_and_hwnd(tw):
    by_title = find_window(title=tw.title)
    assert by_title.hwnd == tw.hwnd
    assert find_window(hwnd=tw.hwnd).title == tw.title
    assert by_title.client[2] > 0 and by_title.client[3] > 0
    assert any(w.hwnd == tw.hwnd for w in list_windows(title="reflex-test"))


def test_missing_window_lists_what_exists():
    with pytest.raises(WindowNotFound, match="no visible window matches"):
        find_window(title="no window is called this 7d1f")


# --------------------------------------------------------------------------- capture


def test_window_capture_is_client_area_with_child(tw):
    src = WindowSource(tw.hwnd)
    frame = src.grab()
    win = find_window(hwnd=tw.hwnd)
    assert frame.shape == (win.client[3], win.client[2], 3)
    assert _near(frame[20, 10], RED_BGR)
    assert _near(frame[20, frame.shape[1] - 10], BLUE_BGR)
    assert src.active_method == "printwindow"


def test_window_capture_works_while_covered(tw):
    _needs_screen()
    green, yellow = (40, 200, 40), (40, 220, 220)
    with TestWindow(title="reflex-cover", pos=tw.pos, size=tw.size, colors=(green, yellow)):
        x, y, cw, ch = find_window(hwnd=tw.hwnd).client
        on_screen = ScreenSource((x, y, cw, ch)).grab()
        assert not _near(on_screen[20, 10], RED_BGR), "tw should be covered for this test to mean anything"
        frame = WindowSource(tw.hwnd).grab()
        assert _near(frame[20, 10], RED_BGR)
        assert _near(frame[20, frame.shape[1] - 10], BLUE_BGR)


def test_window_capture_follows_a_moved_window(tw):
    src = WindowSource(tw.hwnd)
    before = src.grab()
    tw.move(300, 260)
    time.sleep(0.1)
    after = src.grab()
    assert before.shape == after.shape and _near(after[20, 10], RED_BGR)


def test_minimized_window_raises_instead_of_black_frames(tw):
    import ctypes

    ctypes.windll.user32.ShowWindow(tw.hwnd, 7)  # SW_SHOWMINNOACTIVE
    time.sleep(0.1)
    with pytest.raises(CaptureError, match="minimized"):
        WindowSource(tw.hwnd).grab()


def test_closed_window_raises(tw):
    src = WindowSource(tw.hwnd)
    tw.close()
    with pytest.raises(WindowNotFound):
        src.grab()


def test_screen_capture_region_shape():
    _needs_screen()
    frame = ScreenSource((0, 0, 64, 48)).grab()
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8


# --------------------------------------------------------------------------- PostMessage


def test_postmessage_keys_reach_background_window(tw):
    sink = PostMessageSink(tw.hwnd)
    assert sink.press("a", hold_ms=5)
    tw.wait_for(lambda e: e[0] == "keyup")
    kinds = [e[0] for e in tw.snapshot()]
    assert kinds[:3] == ["keydown", "char", "keyup"]
    assert tw.snapshot()[0][2] == 0x41


def test_postmessage_click_routes_to_child_in_child_coordinates(tw):
    sink = PostMessageSink(tw.hwnd)
    frame = WindowSource(tw.hwnd).grab()
    w = frame.shape[1]
    sink.click(30, 40)
    sink.click(w - 30, 50, button="right")
    tw.wait_for(lambda e: e[0] == "rbuttonup")
    ev = {e[0]: e for e in tw.snapshot()}
    assert ev["lbuttondown"][1:2] == ("parent",) and ev["lbuttondown"][3:] == (30, 40)
    child_x = (w - 30) - w // 2
    assert ev["rbuttondown"][1] == "child" and ev["rbuttondown"][3:] == (child_x, 50)


def test_postmessage_type_text_and_scroll(tw):
    sink = PostMessageSink(tw.hwnd)
    sink.type_text("hé")
    sink.scroll(10, 10, clicks=-1)
    tw.wait_for(lambda e: e[0] == "wheel")
    chars = [e[2] for e in tw.snapshot() if e[0] == "char"]
    assert chars == [ord("h"), ord("é")]
    wheel = [e for e in tw.snapshot() if e[0] == "wheel"][0]
    assert ((wheel[2] >> 16) & 0xFFFF) == (-120 & 0xFFFF)


def test_postmessage_release_all_lets_go_of_held_keys(tw):
    sink = PostMessageSink(tw.hwnd)
    sink.key_down("right")
    sink.mouse_down(5, 5)
    sink.release_all()
    tw.wait_for(lambda e: e[0] == "lbuttonup")
    kinds = [e[0] for e in tw.snapshot()]
    assert "keyup" in kinds and "lbuttonup" in kinds


# --------------------------------------------------------------------------- SendInput


def test_sendinput_refuses_when_target_lacks_focus(tw):
    sink = SendInputSink(tw.hwnd)  # the test window is never activated
    assert not sink.target_ok()
    assert sink.press("a", hold_ms=1) is False
    assert sink.click(10, 10) is False
    assert sink.sent == 0


@pytest.mark.skipif(os.environ.get("REFLEX_LIVE_INPUT") != "1", reason="moves the real cursor and takes focus")
def test_sendinput_round_trip(tw):
    from reflex.backends.windows import focus

    focus(tw.hwnd)
    time.sleep(0.2)
    sink = SendInputSink(tw.hwnd)
    assert sink.press("b", hold_ms=10)
    assert sink.click(20, 20)
    tw.wait_for(lambda e: e[0] == "lbuttonup")
    kinds = [e[0] for e in tw.snapshot()]
    assert "keydown" in kinds and "lbuttondown" in kinds
