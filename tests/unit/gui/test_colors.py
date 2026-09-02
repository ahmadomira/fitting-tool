"""Tests for gui.plotting.colors — no QApplication required."""

from gui.plotting.colors import rgba


def test_rgba_appends_alpha_defaulting_to_opaque():
    """rgba widens an RGB triple to RGBA; the default must be fully opaque."""
    assert rgba((31, 119, 180)) == (31, 119, 180, 255)
    assert rgba((255, 0, 0), alpha=128) == (255, 0, 0, 128)
