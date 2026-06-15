"""
utils/startup.py
================
Startup validation for the Voice Tour Guide service.

Validates required environment variables before the service
accepts any requests. Exits immediately if any are missing.
"""

import os
import sys
from utils.logger import get_logger

logger = get_logger(__name__)

# No hard-required vars: without ELEVENLABS_API_KEY the service still starts
# and generation returns text-only/no-audio responses through the backend.
REQUIRED_ENV_VARS: list[str] = []


def validate_env() -> None:
    """
    Check all required environment variables are set.
    Exits the process with code 1 if any are missing.
    """
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

    if missing:
        logger.error(
            "Missing required environment variables — service cannot start",
            extra={"missing": missing},
        )
        sys.exit(1)

    logger.info(
        "Environment validation passed",
        extra={"checked": len(REQUIRED_ENV_VARS)},
    )
