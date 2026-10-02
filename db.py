"""SQLAlchemy engine, session factory, and declarative base."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
