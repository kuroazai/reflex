# reflex

Screen-reading macros and bots for Windows, written in YAML and driven by people or by AI agents.

reflex watches a window, finds images, colours and text in it, and sends keys, clicks and gamepad input back. It can do that while the window is behind other windows and without touching your mouse, so you can keep using the PC. Scripts run step by step ("wait for the Play button, click it, attack until Victory shows"). Watchers react at frame rate ("whenever the heal icon lights up, press 4"). Both work together.

It also runs as an MCP server, so an AI agent can look at a window, act on it, and hand the fast work to a script it writes. An agent takes seconds to think. A boss telegraph lasts 300 ms. reflex is the part that doesn't think.

```
reflex doctor
PASS  window capture (works while covered)  384x261, 76.0 fps, p95 18.5 ms
PASS  background keys (PostMessage)         key down/up delivered without focus
PASS  background clicks (PostMessage)       click routed to the child window under the point
PASS  screen capture                        available
PASS  text recognition (OCR)                Windows OCR (en-GB)
```

## Install

Python 3.10+ on Windows 10 or 11.

```bash
pip install -e ".[all]"     # OCR (the engine built into Windows) and the MCP server
reflex doctor               # proves capture and input work on your machine
```

`pip install -e .` gives you the engine without OCR or MCP. The engine itself is plain Python and runs anywhere. Capture and input are Windows-only.

## Five minutes

Find your window, grab a frame of it, and cut a template out of it:

```bash
reflex windows --title "My Game"
reflex capture --title "My Game" -o shot.png
reflex capture --title "My Game" -o img/play.png --region 600,580,80,40
```

Write a script:

```yaml
name: daily claim
target:
  window: {title: "My Game"}
  input: postmessage            # background input; use sendinput if the game ignores it
steps:
  - wait_for: {image: img/play.png}
    timeout_s: 20
    as: play
  - click: $play
  - wait_for: {text: {contains: "Reward"}}
  - click: {text: {contains: "Claim"}}
  - log: "claimed"
```

Check it, try it without sending input, then run it:

```bash
reflex validate daily.yaml
reflex run daily.yaml --dry-run
reflex run daily.yaml
reflex run daily.yaml --at 09:00 --at 21:00      # or --every 30m
```

**ctrl+alt+end stops any run. ctrl+alt+home pauses it.** Both work from any window, full-screen games included, and stopping releases every key and button reflex was holding.

## Scripts

A script is a list of steps. `reflex schema` prints all 29 with their fields. The ones you'll use most:

| step | what it does |
|---|---|
| `press: space` / `press: [ctrl, s]` | keys in order (`hold_ms`, `times`) |
| `chord: [ctrl, shift, esc]` | keys together |
| `type: "gg ${name}\n"` | text, any characters |
| `click: [x, y]` / `click: $found` / `click: {image: x.png}` | click a point, a stored match, or something it finds now |
| `drag`, `scroll`, `move`, `right_click`, `double_click` | the rest of the mouse |
| `wait: 500` | pause (watchers keep running) |
| `wait_for: <condition>` / `wait_gone: <condition>` | with `timeout_s`, `as`, `on_timeout: fail \| continue \| [steps]` |
| `find`, `read_text` | look once and store the result in a variable |
| `if` / `repeat` | `repeat: {times: N}`, `{while: c}`, `{until: c}`, `{forever: true}`; `break`, `continue` |
| `set`, `call`, `log`, `stop`, `fail`, `screenshot` | variables, functions, output |
| `stick: left` | gamepad stick (see below) |

**Conditions** are screen checks, expressions, or both:

```yaml
{image: img/boss.png}                              # template match (multi-scale, transparent PNGs)
{image: {template: img/coin.png, count_min: 3}}    # at least three on screen
{text: {region: [20, 20, 160, 30], number_below: 300}}   # "HP 240/1200" -> 240 < 300
{color_search: {rgb: [255, 40, 40], region: [0, 0, 1280, 80]}}
{stable: {for_ms: 800}}                            # the screen stopped moving
"${battles < 20 and gold > 500}"
{any_of: [...]}  {all_of: [...]}  {not: ...}
```

Expressions use `${...}`: arithmetic, comparisons, `and`/`or`/`not`, fields of stored matches (`${loot.x + 10}`) and a few functions (`min max abs round int float str len`). They're evaluated against a whitelist, so a macro file can't run code.

Mistakes are caught before the script runs, with their location:

```
FAIL farm.yaml
     farm.yaml: steps[3]<start battle>.clik: a step needs exactly one action, found ['clik'] (did you mean 'click'?)
```

Key names, template files, expressions and function calls are all checked. Runs are bounded too: a script stops at `timeout_s` (default one hour), and every `wait_for` has its own timeout.

`humanize: {jitter_ms: 40, click_jitter_px: 2}` adds small random delays and click offsets. [examples/scripts/auto_battle.yaml](examples/scripts/auto_battle.yaml) uses most of the above.

### Watchers

Watchers are the reactive half: a condition, the input to send, and rules for how often. They run on every frame a script grabs, so a script can be waiting for a loading screen while a watcher closes popups.

```yaml
watchers:
  - id: close_popup
    kind: on_match
    cooldown_s: 1
    params:
      when: {predicate: text, region: [300, 200, 400, 60], contains: "Daily bonus"}
      do: [{click: [680, 215]}]
    expect: {kind: gone, within_ms: 800}     # check the click worked
    trip_after_unverified: 3                 # three misses in a row: stop pressing
```

After each fire, `expect` checks that the screen changed the way it should. If the input keeps doing nothing (wrong window, game lagging), the watcher trips instead of spamming. A watcher file on its own runs with `reflex run watchers.yaml --title "My Game" --duration 30m`.

### Several windows at once

```bash
reflex run farm.yaml --title "Client" --all-windows
```

One copy per matching window, in parallel. Background capture and input are what make this work: no instance needs focus.

## Recording

```bash
reflex record --title "My Game" -o farm.yaml --images
```

Play normally, then press ctrl+alt+end. You get an ordinary script you can edit. Typing becomes one `type` step using your real keyboard layout, held keys keep their hold time, press-move-release becomes a `drag`, and pauses become `wait` steps.

With `--images`, each click also saves a 48 px crop of what was under the pointer and clicks *that* instead of a fixed pixel, so the script survives the window moving. Input reflex sends itself is never recorded.

## For AI agents (MCP)

```bash
claude mcp add reflex -- reflex mcp --base-dir C:/path/to/macros
```

19 tools. `screenshot` returns the window as an image (covered windows too) with the scale to map coordinates back. `find_image`, `find_text`, `read_text` and `find_color` look things up. `click`, `press`, `type_text`, `drag` and `scroll` act, in the background by default. `save_template` turns a region the agent can see into an image condition.

Then the part that matters. `schema`, `validate_script` and `run_script` (or `watch` for watchers) let the agent write a script and run it in the background; the call returns a run id at once. The agent polls `run_status` for the log and result and calls `stop_run` when it's done. `dry_run` reports the input a script *would* have sent.

The tools are plain methods on `reflex.agent.ReflexTools`, so other agent frameworks can call them without MCP.

## Python

```python
from reflex.backends.windows import WindowSource, PostMessageSink, find_window
from reflex.vision import find_image
from reflex.ocr import read_text

win = find_window(title="My Game")
frame = WindowSource(win).grab()              # BGR numpy array of the client area
hits = find_image(frame, "img/coin.png", max_results=10)
hp = read_text(frame, (20, 20, 160, 30)).text
PostMessageSink(win).click(*hits[0].center)
```

Scripts: `reflex.script.load_script()` and `run_script()`. Watchers: `reflex.runtime.ReflexRuntime`. Predicates, recipes and expectations are plug-ins (`reflex list`), and other packages can register their own through the `reflex.plugins` entry point.

## Input: which one to use

| | `postmessage` (default) | `sendinput` |
|---|---|---|
| needs the window focused | no | yes (it waits for focus rather than typing elsewhere) |
| moves your real cursor | no | yes |
| works while you use the PC | yes | no |
| works with | most apps, launchers, emulators, many games | games that read raw input or DirectInput |

If clicks or keys do nothing with `postmessage`, switch to `sendinput`.

**Gamepad:** `target: {gamepad: true}` adds a virtual Xbox 360 controller. `pad_a`, `pad_rt`, `pad_up` and the rest work anywhere a key name does, and `stick: left` with `x`/`y` moves a stick. It needs [vgamepad](https://pypi.org/project/vgamepad/) and the ViGEmBus driver, which you install yourself.

## Limits, and what reflex won't do

- **Minimized windows can't be captured.** There's nothing to render. Covered is fine. reflex raises an error rather than handing your conditions black frames.
- **A locked PC refuses screen copies and real input.** Window capture and background input keep working.
- **A few games render black to window capture.** reflex notices and falls back to copying the screen area, which needs the window visible.
- **reflex sends ordinary Windows input and makes no attempt to hide from anti-cheat.** No kernel drivers, no detection evasion, and that won't change. Many online games ban automation in their terms; check before running a bot on an account you care about.

## Development

```bash
pip install -e ".[all,test]"
pytest                       # 234 tests; the Windows ones drive real windows the tests create
REFLEX_LIVE_INPUT=1 pytest   # also tests SendInput, which moves your cursor
```

The tests never touch your own windows. `reflex.backends.windows.testwindow` makes a real Win32 window that paints known colours and records every message it gets, so capture and input are tested end to end. The replay fixtures are drawn by a script, not captured from a game.

MIT licence.
