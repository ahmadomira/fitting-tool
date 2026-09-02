"""Color constants and helpers for the plotting module."""

from __future__ import annotations

# 8-color cycle for active replicas (RGB) — default "Tab10" style
REPLICA_PALETTE: list[tuple[int, int, int]] = [
    (31, 119, 180),  # blue
    (255, 127, 14),  # orange
    (44, 160, 44),  # green
    (214, 39, 40),  # red
    (148, 103, 189),  # purple
    (140, 86, 75),  # brown
    (227, 119, 194),  # pink
    (127, 127, 127),  # grey
]

# Named palettes for replica coloring (matplotlib-inspired, 8 colors each)
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    'Default (Tab10)': REPLICA_PALETTE,
    'Set1': [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
    ],
    'Set2': [
        (102, 194, 165),
        (252, 141, 98),
        (141, 160, 203),
        (231, 138, 195),
        (166, 216, 84),
        (255, 217, 47),
        (229, 196, 148),
        (179, 179, 179),
    ],
    'Dark2': [
        (27, 158, 119),
        (217, 95, 2),
        (117, 112, 179),
        (231, 41, 138),
        (102, 166, 30),
        (230, 171, 2),
        (166, 118, 29),
        (102, 102, 102),
    ],
    'Paired': [
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),
        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),
    ],
    'Pastel1': [
        (251, 180, 174),
        (179, 205, 227),
        (204, 235, 197),
        (222, 203, 228),
        (254, 217, 166),
        (255, 255, 204),
        (229, 216, 189),
        (253, 218, 236),
    ],
    'Accent': [
        (127, 201, 127),
        (190, 174, 212),
        (253, 192, 134),
        (255, 255, 153),
        (56, 108, 176),
        (240, 2, 127),
        (191, 91, 23),
        (102, 102, 102),
    ],
}

PALETTE_NAMES: list[str] = list(PALETTES.keys())

# Fit curve colors (distinct from replica palette)
FIT_PALETTE: list[tuple[int, int, int]] = [
    (23, 190, 207),  # cyan
    (188, 189, 34),  # yellow-green
    (174, 199, 232),  # light blue
    (255, 187, 120),  # light orange
]

DROPPED_REPLICA_COLOR: tuple[int, int, int] = (180, 180, 180)
AVERAGE_LINE_COLOR: tuple[int, int, int] = (0, 0, 0)
ERROR_BAR_COLOR: tuple[int, int, int] = (80, 80, 80)

BACKGROUND_COLOR = 'w'
FOREGROUND_COLOR = 'k'


def rgba(rgb: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    """Return an RGBA tuple from an RGB tuple and optional alpha.

    Parameters
    ----------
    rgb : tuple[int, int, int]
        Red, green, blue values (0–255).
    alpha : int
        Alpha value (0–255), default 255 (fully opaque).

    Returns
    -------
    tuple[int, int, int, int]
    """
    r, g, b = rgb
    return (r, g, b, alpha)
