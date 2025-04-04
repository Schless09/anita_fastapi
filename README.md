# Anita - AI-Powered Recruitment System

Anita is a sophisticated recruitment system built with FastAPI and LangChain that automates and enhances the recruitment process using AI. The system handles candidate intake, job matching, interview scheduling, and follow-up communications.

## System Architecture

### Core Components

1. **Agents**

   - `CandidateIntakeAgent`: Processes candidate submissions and initial screening
   - `JobMatchingAgent`: Matches candidates to jobs using semantic search
   - `FarmingMatchingAgent`: Proactively matches candidates to new jobs
   - `InterviewAgent`: Manages interview scheduling and feedback
   - `FollowUpAgent`: Handles follow-up communications

2. **Tools**

   - `PDFProcessor`: Extracts text from PDF resumes
   - `ResumeParser`: Parses resume text into structured data
   - `VectorStoreTool`: Manages vector storage for jobs and candidates
   - `MatchingTool`: Handles job-candidate matching logic
   - `EmailTool`: Manages email communications
   - `CalendarTool`: Handles calendar operations

3. **Chains**
   - `CandidateProcessingChain`: Orchestrates candidate intake process
   - `JobMatchingChain`: Orchestrates job matching process
   - `InterviewSchedulingChain`: Orchestrates interview scheduling
   - `FollowUpChain`: Orchestrates follow-up communications

### Data Flow

1. **Candidate Intake**

   - PDF resume upload
   - Text extraction and parsing
   - Profile creation and vector storage
   - Confirmation email

2. **Job Matching**

   - Job posting analysis
   - Semantic matching with candidates
   - Match scoring and ranking
   - Notification emails

3. **Interview Process**

   - Interview scheduling
   - Calendar management
   - Material preparation
   - Feedback processing

4. **Follow-up Management**
   - Response tracking
   - Automated follow-ups
   - Status updates
   - Reporting

## Setup

### Prerequisites

- Python 3.9+ # Check compatibility if using older Python
- FastAPI
- LangChain
- OpenAI API key
- Supabase account credentials
- Retell API key and Agent ID
- Twilio credentials (if used by Retell/directly)
- Slack credentials (if Slack integration used)
- Ngrok account (for local development with webhooks)
- Google OAuth Credentials (if using Gmail for email)

### Environment Variables

Create a `.env` file in the project root and add the following variables based on your `app/config/settings.py`:

```env
# Environment
ENVIRONMENT=development # or production

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo-preview # Or your preferred model
# OPENAI_TEMPERATURE=0.5 # Optional, defaults in settings

# Retell
RETELL_AGENT_ID=your_retell_agent_id
RETELL_API_KEY=your_retell_api_key
RETELL_API_BASE=https://api.retellai.com # Or your custom base
RETELL_FROM_NUMBER=your_retell_phone_number
# RETELL_WEBHOOK_URL is derived automatically

# Twilio
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number

# Slack
SLACK_APP_ID=
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=
SLACK_SIGNING_SECRET=
SLACK_VERIFICATION_TOKEN=
SLACK_BOT_TOKEN=

# Ngrok / Base URL (for local dev)
NGROK_AUTHTOKEN=your_ngrok_token
BASE_URL=your_ngrok_https_url # e.g., https://xxxxx.ngrok-free.app

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Vercel (If using Vercel deployment protection)
Vercel_PROTECTION_BYPASS=your_secret

# Add Gmail / Google OAuth variables here if needed for email
# e.g., GOOGLE_CLIENT_SECRETS_JSON=path/to/credentials.json
# e.g., SENDER_EMAIL=your_authenticated_email@example.com
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/anita_fastapi.git
cd anita_fastapi
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

## Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

1. **Candidate Management**

   - `POST /api/candidates`: Submit a new candidate
   - `GET /api/candidates/{id}`: Get candidate details
   - `PUT /api/candidates/{id}`: Update candidate status

2. **Job Management**

   - `POST /api/jobs`: Create a new job posting
   - `GET /api/jobs/{id}`: Get job details
   - `GET /api/jobs/{id}/matches`: Get matching candidates

3. **Interview Management**

   - `POST /api/interviews`: Schedule an interview
   - `POST /api/interviews/{id}/feedback`: Submit interview feedback
   - `GET /api/interviews/{id}`: Get interview details

4. **Follow-up Management**
   - `POST /api/follow-ups`: Send a follow-up message
   - `GET /api/follow-ups/report`: Get follow-up activity report

### Example Usage

1. **Submit a Candidate**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/candidates",
    files={"resume": open("path/to/resume.pdf", "rb")},
    data={
        "email": "candidate@example.com",
        "name": "John Doe"
    }
)
```

2. **Create a Job Posting**

```python
response = requests.post(
    "http://localhost:8000/api/jobs",
    json={
        "title": "Senior Software Engineer",
        "company": "Tech Corp",
        "description": "Looking for an experienced software engineer...",
        "requirements": ["Python", "FastAPI", "LangChain"]
    }
)
```

3. **Schedule an Interview**

```python
response = requests.post(
    "http://localhost:8000/api/interviews",
    json={
        "candidate_id": "candidate_123",
        "job_id": "job_456",
        "preferred_times": ["2024-03-20T10:00:00Z", "2024-03-20T14:00:00Z"],
        "duration_minutes": 60
    }
)
```

## Development

### Project Structure

```
anita_fastapi/
├── agents/
│   └── langchain/
│       ├── agents/
│       ├── chains/
│       └── tools/
├── services/
│   ├── candidate.py
│   ├── job.py
│   ├── matching.py
│   └── openai.py
├── main.py
├── requirements.txt
└── README.md
```

### Adding New Features

1. Create new agent/tool/chain in the appropriate directory
2. Update the main FastAPI application to include new endpoints
3. Add necessary tests
4. Update documentation

### Testing

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

_Note: Recent updates include migration from Pinecone/SendGrid to Supabase vectors/alternative email and deployment configuration fixes (as of commit xxxxxxx)._
