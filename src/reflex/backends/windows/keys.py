"""Key names → Windows virtual-key codes.

Names are case-insensitive. Single characters that are not in the table
(e.g. "é") fall back to the active keyboard layout via VkKeyScanW.
"""

from __future__ import annotations

from dataclasses import dataclass

_VK: dict[str, int] = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "shift": 0x10, "ctrl": 0x11, "control": 0x11, "alt": 0x12, "pause": 0x13,
    "capslock": 0x14, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
    "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E, "del": 0x2E,
    "lwin": 0x5B, "win": 0x5B, "rwin": 0x5C, "apps": 0x5D, "menu": 0x5D,
    "multiply": 0x6A, "add": 0x6B, "subtract": 0x6D, "decimal": 0x6E, "divide": 0x6F,
    "numlock": 0x90, "scrolllock": 0x91,
    "lshift": 0xA0, "rshift": 0xA1, "lctrl": 0xA2, "rctrl": 0xA3, "lalt": 0xA4, "ralt": 0xA5,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF, "`": 0xC0,
    "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    _VK[_c] = ord(_c.upper())
for _d in "0123456789":
    _VK[_d] = ord(_d)
    _VK[f"num{_d}"] = 0x60 + int(_d)
    _VK[f"numpad{_d}"] = 0x60 + int(_d)
for _n in range(1, 25):
    _VK[f"f{_n}"] = 0x6F + _n

# Keys that live on the extended part of the keyboard: SendInput and the
# WM_KEY* lParam both need the extended bit or the game sees the numpad twin.
EXTENDED_VK = frozenset({
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  # page/home/end/arrows
    0x2C, 0x2D, 0x2E,  # printscreen/insert/delete
    0x5B, 0x5C, 0x5D,  # win/apps
    0x6F, 0x90,  # divide, numlock
    0xA3, 0xA5,  # rctrl, ralt
})

MOUSE_BUTTONS = ("left", "right", "middle")


class UnknownKey(ValueError):
    pass


@dataclass(frozen=True)
class KeyCode:
    name: str
    vk: int
    scan: int
    extended: bool


def vk_for(name: str) -> int:
    key = str(name).strip()
    low = key.lower()
    if low in _VK:
        return _VK[low]
    if low.startswith("vk:") or low.startswith("0x"):
        return int(low[3:] if low.startswith("vk:") else low, 16 if low.startswith("0x") else 10)
    if len(key) == 1:
        from reflex.backends.windows._win32 import user32

        res = int(user32.VkKeyScanW(key))
        if res != -1:
            return res & 0xFF
    raise UnknownKey(f"unknown key {name!r}; known: {', '.join(sorted(known_keys()))}")


def resolve(name: str) -> KeyCode:
    from reflex.backends.windows._win32 import MAPVK_VK_TO_VSC, user32

    vk = vk_for(name)
    scan = int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC))
    return KeyCode(name=str(name), vk=vk, scan=scan, extended=vk in EXTENDED_VK)


def known_keys() -> list[str]:
    return sorted(_VK)


def key_lparam(scan: int, *, extended: bool, up: bool, alt: bool = False) -> int:
    """lParam for WM_KEYDOWN / WM_KEYUP (repeat 1, scan code, flags)."""
    lp = 1 | ((scan & 0xFF) << 16)
    if extended:
        lp |= 1 << 24
    if alt:
        lp |= 1 << 29
    if up:
        lp |= (1 << 30) | (1 << 31)
    return lp
