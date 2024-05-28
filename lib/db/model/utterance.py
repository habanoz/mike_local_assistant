from datetime import datetime

from sqlalchemy import Column, DateTime, String, UUID, JSON

from lib.db.base import Base


class Utterance(Base):
    __tablename__ = 'utterances'
    id = Column(UUID(as_uuid=True), primary_key=True)
    chat_id = Column(UUID(as_uuid=True))
    type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    files = Column(JSON, nullable=True)
    debug = Column(JSON, nullable=True)
    alternatives = Column(JSON, nullable=True)
    created = Column(DateTime, nullable=False, default=datetime.now)

    def __repr__(self):
        return f"<Utterance(id={self.id}, chat_id={self.chat_id}, type={self.type})>"

    def __eq__(self, other):
        if isinstance(other, Utterance):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
