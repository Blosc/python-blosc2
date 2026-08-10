#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Waiting for a Textual app to settle, shared by the b2view suites.

``pilot.press`` delivers the key, but the app acts on it over the frames that
follow: pushing a screen, rebuilding a widget, reloading a page.  Asserting
straight after the press is therefore a race, and one that an idle developer
machine wins and a loaded CI runner loses -- which is how it reaches us, as a
single test failing on one platform.

A fixed number of ``pilot.pause()`` calls is the same race with a bigger
constant; the ``await pilot.pause()`` doubled up before some assertions here is
what that looks like in the source.  Poll for the state instead, with a bound
so a genuine hang still fails rather than hanging the suite.

Each b2view module had grown its own ``_wait_table``-shaped helper; this is the
piece worth having in one place, since the races are identical across them.
"""

from __future__ import annotations

#: How many frames to pump before giving up.  Generous: the cost of a high
#: bound is paid only when something is genuinely broken, while the cost of a
#: low one is a flake on whichever runner happens to be slowest that day.
MAX_FRAMES = 100


async def wait_until(pilot, predicate, *, message="condition not met in time") -> None:
    """Pump the event loop until *predicate* holds, or fail."""
    for _ in range(MAX_FRAMES):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError(message)


async def wait_for_screen(pilot, screen_cls, *, present: bool = True):
    """Wait until the app's active screen is (or is no longer) *screen_cls*.

    Returns the active screen, so a caller that goes on to query widgets on it
    reads the same object the wait settled on.
    """
    what = screen_cls.__name__

    def settled():
        return isinstance(pilot.app.screen, screen_cls) is present

    await wait_until(
        pilot,
        settled,
        message=(f"screen never became {what}" if present else f"screen never left {what}")
        + f" (still {type(pilot.app.screen).__name__})",
    )
    return pilot.app.screen
