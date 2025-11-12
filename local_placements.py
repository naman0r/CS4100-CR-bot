"""
local_placements.py

This file is intended for each user to specify their own
Clash Royale bot placement coordinates locally. 
Do NOT push this file to GitHub; add it to .gitignore.
"""

# -------------------
# Card slots (hand)
# -------------------
# Order: leftmost card to rightmost card
card_slots = [
    (900, 900),  # Slot 1
    (1000, 900), # Slot 2
    (1100, 900), # Slot 3
    (1200, 900)  # Slot 4
]

# -------------------
# Arena placements
# -------------------

# Behind King Tower (defensive placements)
king_tower = {
    "left": (965, 777),
    "right": (1035, 777) 
}

# In front of Princess Towers (defensive/offensive)
princess_towers = {
    "left": (860, 610),
    "right": (1140, 610)
}

# Bridge placements (offensive drop zones)
bridge = {
    "left": (860, 470),
    "right": (1140, 470)
}

# -------------------
# Optional arena zones for random placement
# -------------------
arena_zone = {
    "x": (800, 1200),
    "y": (480, 750)
}

# -------------------
# Usage in bot code:
# from local_placements import card_slots, king_tower, princess_towers, bridge, arena_zone
# -------------------
