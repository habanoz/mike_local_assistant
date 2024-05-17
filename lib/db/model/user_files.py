from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text, UUID, UniqueConstraint
from sqlalchemy.orm import deferred

from lib.db.base import Base


class UserFile(Base):
    __tablename__ = 'user_files'
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    content = deferred(Column(Text, nullable=False))
    summary = Column(String, nullable=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    UniqueConstraint('name', name='file_name__user_files__uq_ind')

    def __repr__(self):
        return f"<UserFile(name={self.name}, id={self.id})>"

    def __eq__(self, other):
        if isinstance(other, UserFile):
            return self.id == other.id
        return False
