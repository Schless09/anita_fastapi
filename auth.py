from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import os

# Ensure credentials.json is in the same directory as this script
CREDENTIALS_PATH = 'credentials.json'
TOKEN_PATH = 'token.pkl'
SCOPES = ['https://www.googleapis.com/auth/gmail.send'] # Make sure this matches the scope needed by your app

def main():
    """Runs the OAuth 2.0 flow and saves the credentials."""
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Error: {CREDENTIALS_PATH} not found. Please place the credentials file in the script's directory.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    # port=0 finds a random available port
    print("Launching browser for authentication...")
    creds = flow.run_local_server(port=0)

    # Save the credentials for the next run
    with open(TOKEN_PATH, 'wb') as token:
        pickle.dump(creds, token)

    print(f"✅ Auth successful! Token saved as {TOKEN_PATH}")

if __name__ == '__main__':
    main() 