#!/bin/bash

# Ensure uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Start the GUI
echo "Starting Video Mixer GUI..."
uv run python gui_main.py