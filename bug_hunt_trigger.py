#!/usr/bin/env python3
"""
Cron-triggered script: posts a bug-hunt prompt to the
#inoculation-bootstrap-heuristic Slack channel as a new top-level thread.

Runs every day at 06:00 UK time (TZ=Europe/London in crontab).
"""

import os
import sys
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

CHANNEL_ID = "C0AJAR3TC92"

MESSAGE = (
    "<@U0AHV88PY81> Search for critical bugs that are dangerous to the validity of the "
    "experiments. List them without fixing them yet."
)


def main() -> None:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("ERROR: SLACK_BOT_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    client = WebClient(token=token)
    try:
        response = client.chat_postMessage(
            channel=CHANNEL_ID,
            text=MESSAGE,
        )
        print(f"Posted successfully — ts={response['ts']}")
    except SlackApiError as exc:
        print(f"Slack API error: {exc.response['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
