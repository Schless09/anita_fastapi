import requests
import json

# Test webhook URL
WEBHOOK_URL = "http://localhost:8000/webhook/retell"

# Sample webhook payload
payload = {
    "event": "call_ended",
    "call": {
        "call_id": "test_call_123",
        "call_status": "ended",
        "metadata": {
            "candidate_id": "test_candidate_123"
        },
        "transcript": "Hi, this is a test transcript.",
        "transcript_object": [
            {
                "speaker": "agent",
                "text": "Hi, how are you today?"
            },
            {
                "speaker": "candidate",
                "text": "I'm doing well, thank you."
            }
        ],
        "call_analysis": {
            "call_summary": "Test call summary",
            "user_sentiment": "positive",
            "call_successful": True
        },
        "recording_url": "https://example.com/recording.mp3",
        "start_timestamp": "2024-03-19T10:00:00Z",
        "end_timestamp": "2024-03-19T10:15:00Z",
        "call_type": "initial_contact",
        "agent_id": "test_agent_123"
    }
}

# Headers (without signature since Retell will add this)
headers = {
    "Content-Type": "application/json"
}

# Make request
print("\n🚀 Sending test webhook request...")
print("\n📦 Payload:")
print(json.dumps(payload, indent=2))

response = requests.post(WEBHOOK_URL, json=payload, headers=headers)

print("\n📬 Response:")
print(f"Status Code: {response.status_code}")
try:
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
except:
    print(f"Response Body: {response.text}") 