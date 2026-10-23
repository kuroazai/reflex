"""Platform backends for screen capture and input.

`reflex.backends.windows` is the only one today. The engine itself is
platform-neutral: anything with `grab()` is a frame source and anything with
`press()` / `click()` is an input sink.
"""
