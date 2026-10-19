"""Compatibility entry point for the research-hardened Stage 4 protocol.

Model selection, calibration, threshold selection, locked-test evaluation,
and acceptance gating live in ``harden_emotion_model.py`` so there is only
one reproducible training path.
"""

from harden_emotion_model import main


if __name__ == "__main__":
    main()
