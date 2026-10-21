"""Append-only audit log of every step a complaint goes through.

Only masked text is ever written here. Postgres in production
(DATABASE_URL=postgresql+psycopg://...), SQLite for local runs and tests.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src import config


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    complaint_id: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(128), default="system")
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AuditLog:
    def __init__(self, url=config.DATABASE_URL):
        if url.startswith("sqlite:///"):
            config.DATA_DIR.mkdir(exist_ok=True)
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)

    def log(self, complaint_id, step, payload, actor="system", model_version=None):
        with Session(self.engine) as session:
            session.add(AuditEvent(complaint_id=complaint_id, step=step, actor=actor,
                                   model_version=model_version, payload=payload))
            session.commit()

    def events(self, complaint_id=None, step=None):
        query = select(AuditEvent).order_by(AuditEvent.id)
        if complaint_id:
            query = query.where(AuditEvent.complaint_id == complaint_id)
        if step:
            query = query.where(AuditEvent.step == step)
        with Session(self.engine) as session:
            return [
                {"complaint_id": e.complaint_id, "step": e.step, "actor": e.actor,
                 "model_version": e.model_version, "payload": e.payload,
                 "created_at": e.created_at.isoformat()}
                for e in session.scalars(query)
            ]
