#!/usr/bin/env python3
"""
Setup Google Drive authentication with write access for PDFScribe.
Run this once to enable uploading transcriptions to Google Drive.
"""
import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Scopes needed for PDFScribe (read + write to Drive)
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',  # Create/edit files created by app
    'https://www.googleapis.com/auth/drive.readonly',  # Read all files
]

OAUTH_PATH = os.path.expanduser("~/.config/mcp-gdrive/gcp-oauth.keys.json")
CREDS_PATH = os.path.expanduser("~/.config/mcp-gdrive/.gdrive-server-credentials.json")

def main():
    print("PDFScribe Google Drive Authentication Setup")
    print("=" * 50)
    print(f"\nThis will update credentials at:\n  {CREDS_PATH}")
    print(f"\nRequested scopes:\n  - drive.file (create/edit app files)")
    print(f"  - drive.readonly (read all files)")

    if not os.path.exists(OAUTH_PATH):
        print(f"\nERROR: OAuth keys not found at {OAUTH_PATH}")
        print("Please set up Google OAuth credentials first.")
        return

    # Load existing credentials if any
    creds = None
    if os.path.exists(CREDS_PATH):
        with open(CREDS_PATH) as f:
            token_data = json.load(f)

        with open(OAUTH_PATH) as f:
            oauth_keys = json.load(f)['installed']

        creds = Credentials(
            token=token_data.get('access_token'),
            refresh_token=token_data.get('refresh_token'),
            token_uri='https://oauth2.googleapis.com/token',
            client_id=oauth_keys['client_id'],
            client_secret=oauth_keys['client_secret'],
            scopes=token_data.get('scope', '').split()
        )

    # Check if we need new credentials
    needs_reauth = True
    if creds and creds.valid:
        existing_scopes = set(creds.scopes or [])
        needed_scopes = set(SCOPES)
        if needed_scopes.issubset(existing_scopes):
            print("\n✓ Credentials already have required scopes!")
            needs_reauth = False

    if needs_reauth:
        print("\n→ Opening browser for authentication...")
        print("  Please sign in and grant the requested permissions.\n")

        # Create flow from OAuth keys
        flow = InstalledAppFlow.from_client_secrets_file(OAUTH_PATH, SCOPES)
        creds = flow.run_local_server(port=8080)

        # Save credentials in the format expected by the MCP server
        token_data = {
            'access_token': creds.token,
            'refresh_token': creds.refresh_token,
            'scope': ' '.join(creds.scopes),
            'token_type': 'Bearer',
            'expiry_date': int(creds.expiry.timestamp() * 1000) if creds.expiry else None
        }

        with open(CREDS_PATH, 'w') as f:
            json.dump(token_data, f, indent=2)

        print(f"\n✓ Credentials saved to {CREDS_PATH}")
        print("✓ PDFScribe can now upload transcriptions to Google Drive!")

if __name__ == "__main__":
    main()
