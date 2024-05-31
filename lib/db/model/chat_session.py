from datetime import datetime

from sqlalchemy import Column, DateTime, UUID

from lib.db.base import Base


class ChatSession(Base):
    __tablename__ = 'chats'
    id = Column(UUID(as_uuid=True), primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.now)
    update = Column(DateTime, nullable=False, default=datetime.now)

    def __eq__(self, other):
        if isinstance(other, ChatSession):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
