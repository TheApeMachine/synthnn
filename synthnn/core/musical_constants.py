"""Shared musical constants for SynthNN."""

# Mapping of musical modes to frequency ratios.
MODE_INTERVALS = {
    'ionian': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2],
    'dorian': [1, 9/8, 6/5, 4/3, 3/2, 5/3, 9/5, 2],
    'phrygian': [1, 16/15, 6/5, 4/3, 3/2, 8/5, 9/5, 2],
    'lydian': [1, 9/8, 5/4, 45/32, 3/2, 5/3, 15/8, 2],
    'mixolydian': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 16/9, 2],
    'aeolian': [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2],
    'locrian': [1, 16/15, 6/5, 4/3, 64/45, 8/5, 9/5, 2],
}

# Basic roman numeral chord mapping used for quick chord generation.
ROMAN_CHORD_MAP = {
    'I': [1, 5/4, 3/2],
    'IV': [4/3, 1, 5/4],
    'V': [3/2, 5/4, 9/8],
}
