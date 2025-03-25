# Anita FastAPI - AI-Powered Recruitment Platform

An intelligent recruitment platform built with FastAPI that automates candidate communication and job matching using AI. The system features advanced email monitoring, smart response generation, and seamless candidate-job matching.

## Features

### Email Communication System

- **Automated Email Monitoring**: Continuously monitors and processes incoming candidate emails
- **Smart Response Generation**: Automatically answers candidate questions about job details
- **Thread Tracking**: Maintains conversation context through email thread tracking
- **Human Fallback**: Forwards complex queries to human support when needed

### Candidate Management

- **Profile Processing**: Extracts and structures candidate information from conversations
- **Intelligent Matching**: Matches candidates with suitable job positions
- **Preference Tracking**: Maintains candidate preferences for better job recommendations
- **Asynchronous Resume Processing**: Background processing of candidate resumes with detailed error tracking
- **Enhanced Logging**: Comprehensive logging system for better debugging and monitoring

### Job Management

- **Structured Job Data**: Maintains detailed job information including requirements, benefits, and tech stack
- **Vector Search**: Enables semantic search for matching candidates with positions
- **Real-time Updates**: Keeps job information current for accurate candidate communications

## Technical Stack

- **Backend Framework**: FastAPI
- **Email Service**: SendGrid (both outbound and inbound parsing)
- **Vector Store**: Pinecone for semantic search
- **AI Integration**: OpenAI GPT-4 for natural language processing
- **Data Processing**: Python with advanced regex for email parsing
- **Asynchronous Processing**: FastAPI Background Tasks for non-blocking operations

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/anita_fastapi.git
cd anita_fastapi
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration:
# - SENDGRID_API_KEY
# - SENDER_EMAIL
# - OPENAI_API_KEY
# - SENDGRID_INBOUND_HOSTNAME
# - RETELL_API_KEY
# - RETELL_AGENT_ID
# - RETELL_FROM_NUMBER
```

4. Configure SendGrid:

- Set up SendGrid Inbound Parse webhook
- Configure domain authentication
- Set up email forwarding

5. Start the server:

```bash
uvicorn main:app --reload --log-level debug
```

## API Endpoints

### Email Handling

- `POST /email/webhook`: Handles incoming emails from SendGrid
- `POST /test-email`: Sends a test email to verify setup

### Call Management

- `/makeCall` - Trigger a new call with Retell AI
- `/webhook` - Handle Retell AI webhook events
- `/cleanup` - Clean up completed calls and process transcripts
- `/transcript/{call_id}` - Get transcript for a specific call
- `/status/{call_id}` - Get status for a specific call

### Candidate Management

- `POST /candidates`: Submit a new candidate profile
- `GET /candidates/{candidate_id}/profile`: Get candidate profile
- `POST /candidates/match-jobs`: Find matching jobs for a candidate
- `POST /candidate/retell-transcript`: Process and store call transcripts
- `POST /api/makeCall`: Initiate AI-powered candidate calls
- `/candidates` - Get all candidate profiles
- `/candidates/{candidate_id}` - Get a specific candidate profile
- `/candidates/{candidate_id}/transcript` - Get transcript for a specific candidate

### Job Management

- `POST /jobs/submit`: Submit new job posting
- `GET /jobs/open-positions`: List all open positions
- `POST /jobs/match-candidates`: Find matching candidates for a job
- `GET /jobs/most-recent`: Get the most recent job posting
- `GET /jobs/{job_id}`: Get a specific job posting
- `GET /jobs/{job_id}/status`: Get the status of a job posting

### System Management

- `POST /webhook/retell`: Handle Retell AI call status updates

## Development

### Branch Strategy

- `main`: Production-ready code
- `feature/*`: New feature development
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Testing

Run tests with:

```bash
pytest
```

### Debugging

The system includes comprehensive logging for debugging:

- Background task processing logs
- API endpoint request/response logs
- Error tracking with full stack traces
- Process status monitoring

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
