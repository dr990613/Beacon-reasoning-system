def hydrate_time(nanoseconds, tz=None):
    from datetime import time
    nanoseconds_in_day = 24 * 60 * 60 * 1_000_000_000
    nanoseconds = nanoseconds % nanoseconds_in_day
    total_seconds = nanoseconds // 1_000_000_000
    remaining_nanoseconds = nanoseconds % 1_000_000_000
    microseconds = remaining_nanoseconds // 1000
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    return time(hours, minutes, seconds, microseconds, tz)