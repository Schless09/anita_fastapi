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

- Python 3.8+
- FastAPI
- LangChain
- OpenAI API key
- Pinecone API key
- SendGrid API key
- Google Calendar API credentials
- Supabase account

### Environment Variables

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_JOBS_INDEX=your_jobs_index_name
PINECONE_CANDIDATES_INDEX=your_candidates_index_name

# SendGrid
SENDGRID_API_KEY=your_sendgrid_api_key
SENDGRID_SENDER_EMAIL=your_sender_email

# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS=path_to_credentials.json

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
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
