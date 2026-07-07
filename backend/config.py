"""Loads runtime settings from .env — the only place provider choice is wired in."""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_site_url: str
    llm_site_name: str
    host: str
    port: int


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "openai"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "gpt-5.4-nano"),
        llm_site_url=os.environ.get("LLM_SITE_URL", ""),
        llm_site_name=os.environ.get("LLM_SITE_NAME", ""),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
