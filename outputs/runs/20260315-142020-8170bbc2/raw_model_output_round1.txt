def hydrate_time(nanoseconds, tz=None):
    def hydrate_time(nanoseconds, tz=None):
        from datetime import time
        from dateutil.tz import tzutc

        if nanoseconds is None:
            return None

        # Convert nanoseconds to seconds and remaining nanoseconds
        total_seconds = nanoseconds // 1_000_000_000
        remainder_nanos = nanoseconds % 1_000_000_000

        # Split into hours, minutes, seconds
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        # Create a time object with timezone info if provided
        if tz is not None:
            return time(hour=hours, minute=minutes, second=seconds,
                        microsecond=remainder_nanos // 1000, tzinfo=tz)
        else:
            return time(hour=hours, minute=minutes, second=seconds,
                        microsecond=remainder_nanos // 1000)