"""Compatibility entry point; use src/track.py for new commands."""

from track import parse_args, run


if __name__ == "__main__":
    run(parse_args())
