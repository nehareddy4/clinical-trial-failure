"""Connection helper for the AACT cloud Postgres database.

"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_engine():
    user = os.getenv("AACT_USER")
    password = os.getenv("AACT_PASSWORD")
    if not user or not password:
        raise SystemExit(
            "Missing AACT_USER / AACT_PASSWORD. Copy .env.example to .env and fill in "
            "the username and password from your AACT account."
        )
    url = URL.create(
        "postgresql+psycopg2",
        username=user,
        password=password,
        host=os.getenv("AACT_HOST", "aact-db.ctti-clinicaltrials.org"),
        port=int(os.getenv("AACT_PORT", "5432")),
        database=os.getenv("AACT_DB", "aact"),
    )
    return create_engine(url, connect_args={"connect_timeout": 20})
