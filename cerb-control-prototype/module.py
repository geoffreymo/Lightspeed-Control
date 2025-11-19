import random

def random_celestial_emoji():
    """Returns a random celestial emoji."""
    emojis = [
        "🌞",  # Sun
        "🌝",  # Full Moon
        "🌚",  # New Moon
        "🌍",  # Earth
        "🌟",  # Star
        "🌠",  # Shooting Star
        "🌌",  # Milky Way
    ]
    return random.choice(emojis)
