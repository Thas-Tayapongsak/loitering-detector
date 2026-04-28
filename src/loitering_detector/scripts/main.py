import argparse
import logging
import os
import signal
import sys

import redis

from loitering_detector.config import get_config

# Shared Error Classification
INFRA_ERRORS = (
    redis.exceptions.ConnectionError,
    redis.exceptions.TimeoutError,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Loitering Detection System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command to run"
    )

    # Run Sub-command (Headless)
    run_parser = subparsers.add_parser("run", help="Production headless detection loop")
    run_parser.add_argument(
        "--config", default="system_config.yml", help="Path to config"
    )

    # Debug Sub-command (UI)
    debug_parser = subparsers.add_parser(
        "debug", help="Local development with UI visualization"
    )
    debug_parser.add_argument(
        "--config", default="system_config.yml", help="Path to config"
    )
    return parser.parse_args()


def shutdown_handler(signum, frame):
    """Unified handler for SIGINT and SIGTERM."""
    signame = signal.Signals(signum).name
    logging.info("\n%s received. Shutting down gracefully...", signame)
    sys.exit(0)


def _log_infra_error(e, config):
    """Shared infrastructure error reporting."""
    logging.critical("Infrastructure failure: %s", e)
    logging.info(
        "Check Redis at %s:%s", config.loitering.redis.host, config.loitering.redis.port
    )


def main():
    args = parse_args()

    # Shared Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Signal Registration
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Command-Specific Environment Checks
    if args.command == "debug":
        if (os.name != "nt" or sys.platform != "darwin") and not os.environ.get(
            "DISPLAY"
        ):
            logging.critical(
                "No GUI environment detected. Use 'run' for headless mode."
            )
            sys.exit(1)

    # Shared Configuration Loading
    try:
        config = get_config(args.config)
    except Exception as e:
        logging.critical("Failed to load configuration: %s", e)
        sys.exit(1)

    # Dispatch Execution
    try:
        if args.command == "run":
            from loitering_detector.scripts.run import run as run_headless

            run_headless(config)
        elif args.command == "debug":
            from loitering_detector.scripts.debug import run as run_debug

            run_debug(config)
    except INFRA_ERRORS as e:
        _log_infra_error(e, config)
        sys.exit(1)
    except SystemExit:
        pass
    except Exception as e:
        logging.critical("Unexpected system failure: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
