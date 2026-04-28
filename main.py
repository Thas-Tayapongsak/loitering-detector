from loitering_detector.scripts.main import main

# USAGE INSTRUCTIONS:
# ------------------
# To run in production (headless):
#   1. Install headless dependencies:  uv sync --extra headless
#   2. Run:                            uv run main.py run
#   3. Custom config:                  uv run main.py run --config your_config.yml
#
# To run in debug mode (local UI):
#   1. Install GUI dependencies:       uv sync --extra gui
#   2. Run:                            uv run main.py debug
#   3. Custom config:                  uv run main.py debug --config your_config.yml
#
# Console script (after install):
#   loitering-detector run
#   loitering-detector debug --config your_config.yml
#
# REGARDING REDIS:
#   Please ensure Redis is running before starting the application:
#   -> docker compose up -d redis
#   To stop Redis:
#   -> docker compose stop redis


if __name__ == "__main__":
    main()
