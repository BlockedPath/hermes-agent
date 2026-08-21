"""Regression tests for arrow-key decoding in the curses menus.

Root cause these guard against: on many terminals/terminfo entries, cursor
keys are delivered to ``getch()`` as raw CSI/SS3 escape byte sequences
(``27, 91, 66`` for arrow-down) even when ``keypad(True)`` is set. The menus
used to treat the leading ``27`` as ESC/cancel, which dumped the setup wizard's
provider/model picker into its numbered "Select [1-N]" fallback the instant a
user pressed up or down.
"""
import sys

import pytest

# curses (and its _curses C extension) is Unix-only; skip the whole module on Windows.
if sys.platform == "win32":
    pytest.skip("curses is not available on Windows", allow_module_level=True)
import curses

from hermes_cli.curses_ui import (
    _decode_menu_key,
    NAV_CANCEL,
    NAV_DOWN,
    NAV_NONE,
    NAV_SELECT,
    NAV_UP,
    read_menu_key,
)


class FakeStdscr:
    """Minimal stdscr stand-in that replays a queue of getch() byte returns.

    ``getch`` pops from ``keys``; an empty queue yields ``-1`` (matching curses
    non-blocking behavior). ``timeout`` is recorded but otherwise inert.
    """

    def __init__(self, keys):
        self.keys = list(keys)
        self.timeouts = []

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def timeout(self, ms):
        self.timeouts.append(ms)




def test_raw_ss3_arrow_keys_decode():
    # Application cursor mode: ESC O B / ESC O A
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("B")])) == NAV_DOWN
    assert read_menu_key(FakeStdscr([27, ord("O"), ord("A")])) == NAV_UP






def test_enter_variants_select():
    assert read_menu_key(FakeStdscr([10])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([13])) == NAV_SELECT
    assert read_menu_key(FakeStdscr([curses.KEY_ENTER])) == NAV_SELECT






class TestEscapeTailNeverBlocks:
    """Regression for #35: after ESC, the CSI introducer/tail reads used to
    run in RESTORED BLOCKING mode — a bare ESC whose continuation bytes never
    arrived froze the menu indefinitely. Every tail read must stay under the
    short timeout, and blocking mode must be restored on every exit path."""

    class BlockingFakeStdscr:
        """getch() returns queued bytes, then -1 forever (timeout exhausted)."""

        def __init__(self, keys):
            self.keys = list(keys)
            self.timeouts = []
            self.getch_calls = 0

        def timeout(self, ms):
            self.timeouts.append(ms)

        def getch(self):
            self.getch_calls += 1
            return self.keys.pop(0) if self.keys else -1

    def test_bare_esc_then_csi_intro_no_tail_does_not_hang(self):
        # ESC delivered alone in one write; '[' arrives but nothing follows.
        # Old code: final getch() ran BLOCKING -> infinite hang. New code:
        # bounded by the short timeout, returns without hanging.
        scr = self.BlockingFakeStdscr([ord("[")])
        result = _decode_menu_key(scr, 27)
        assert result == NAV_NONE
        assert scr.getch_calls == 2  # bounded: introducer read only
        assert scr.timeouts[-1] == -1  # blocking mode restored

    def test_lone_esc_restores_blocking(self):
        scr = self.BlockingFakeStdscr([])
        assert _decode_menu_key(scr, 27) == NAV_CANCEL
        assert scr.getch_calls == 1
        assert scr.timeouts[-1] == -1

    def test_split_arrow_sequence_still_decodes(self):
        scr = self.BlockingFakeStdscr([ord("["), ord("A")])
        assert _decode_menu_key(scr, 27) == NAV_UP
        assert scr.timeouts[-1] == -1

    def test_csi_tail_with_terminator_bounded(self):
        # Delete key: ESC [ 3 ~ — '~' (0x7E) terminates the tail loop.
        scr = self.BlockingFakeStdscr([ord("["), ord("3"), ord("~")])
        assert _decode_menu_key(scr, 27) == NAV_NONE
        assert scr.getch_calls == 3  # introducer + '3' + terminator '~'
        assert scr.timeouts[-1] == -1

    def test_truncated_csi_tail_self_terminates_on_timeout(self):
        # ESC [ 3 then nothing: tail loop must exit on -1 (outside 0x20..0x3F).
        scr = self.BlockingFakeStdscr([ord("["), ord("3")])
        assert _decode_menu_key(scr, 27) == NAV_NONE
        assert scr.getch_calls == 3
        assert scr.timeouts[-1] == -1

    def test_esc_other_byte_swallow_restores_blocking(self):
        scr = self.BlockingFakeStdscr([ord("z")])
        assert _decode_menu_key(scr, 27) == NAV_NONE
        assert scr.getch_calls == 1
        assert scr.timeouts[-1] == -1
