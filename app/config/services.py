from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI, AsyncOpenAI
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.discovery import build
# import pickle
import os
from functools import lru_cache

from .settings import get_settings

settings = get_settings()

@lru_cache()
def get_openai_client() -> ChatOpenAI:
    """Get OpenAI client instance."""
    return ChatOpenAI(
        model_name=settings.openai_model,
        temperature=settings.openai_temperature
    )

@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings()

# @lru_cache()
# def get_google_calendar_client():
#     """Get Google Calendar client instance."""
#     creds = None
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 settings.google_calendar_credentials,
#                 ['https://www.googleapis.com/auth/calendar']
#             )
#             creds = flow.run_local_server(port=0)
#         
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#     
#     return build('calendar', 'v3', credentials=creds) 