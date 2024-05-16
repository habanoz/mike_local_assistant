from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text, UniqueConstraint, UUID, Integer
from sqlalchemy.orm import deferred

from lib.db.base import Base


class FileChunk(Base):
    __tablename__ = 'file_chunks'
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    idx = Column(Integer, nullable=False)
    chunk_id = Column(String, nullable=False)
    content = deferred(Column(Text, nullable=True))
    created = Column(DateTime, nullable=False, default=datetime.utcnow)
    UniqueConstraint('name', 'idx', name='file_name_idx__file_chunks__uq_ind')

    def __eq__(self, other):
        if isinstance(other, FileChunk):
            return self.id == other.id
        return False
