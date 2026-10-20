"""OAuth flow + YouTube Data API / YouTube Analytics API client factory.

One-time setup: put your own Google Cloud OAuth "Desktop app" credentials in
client_secret.json, then run:
    python src/youtube_auth.py
A browser opens; log into the channel's account and click Allow. token.json
(with a refresh token) is written so later runs need no browser.

A second channel gets its own token: `python src/youtube_auth.py duds_edits`
writes token_duds_edits.json (pick that Brand channel on Google's account
chooser). Pass channel="duds_edits" to get_youtube / get_analytics to use it.
"""
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parent.parent
CLIENT_SECRET = ROOT / "client_secret.json"
TOKEN = ROOT / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def token_path(channel=None):
    return TOKEN if not channel else ROOT / f"token_{channel}.json"


def _token_scopes(channel=None):
    try:
        return set(json.loads(token_path(channel).read_text(encoding="utf-8-sig")).get("scopes") or [])
    except (OSError, json.JSONDecodeError):
        return set()


def _get_credentials(interactive=True, channel=None):
    TOKEN = token_path(channel)
    creds = None
    # from_authorized_user_file does NOT notice a scope upgrade (only expiry /
    # revocation), so an old token.json from before yt-analytics.readonly was
    # added would silently keep working with the narrower scope. Compare
    # explicitly and force a fresh login if anything is missing.
    if TOKEN.exists():
        missing = set(SCOPES) - _token_scopes(channel)
        if missing:
            print(f"token.json is missing scopes {sorted(missing)} - a new login is needed.")
        else:
            creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN.write_text(creds.to_json(), encoding="utf-8")
            return creds
        except Exception as e:  # noqa: BLE001 - revoked/expired refresh token
            print(f"Token refresh failed ({e}); a new login is needed.")
    if not interactive:
        raise RuntimeError("YouTube login required: run `python src/youtube_auth.py` once.")
    if not CLIENT_SECRET.exists():
        raise FileNotFoundError(
            "client_secret.json not found in the project root. Create an OAuth Desktop-app "
            "client in Google Cloud Console (YouTube Data API v3 enabled) and download it there.")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def is_authorized(channel=None):
    return token_path(channel).exists() and not (set(SCOPES) - _token_scopes(channel))


def get_youtube(interactive=True, channel=None):
    return build("youtube", "v3", credentials=_get_credentials(interactive, channel), cache_discovery=False)


def get_analytics(interactive=True, channel=None):
    return build("youtubeAnalytics", "v2", credentials=_get_credentials(interactive, channel),
                 cache_discovery=False)


if __name__ == "__main__":
    import sys
    yt = get_youtube(channel=sys.argv[1] if len(sys.argv) > 1 else None)
    me = yt.channels().list(part="snippet", mine=True).execute()
    items = me.get("items", [])
    print("Authorized for channel:", items[0]["snippet"]["title"] if items else "(no channel found)")
