import uuid
from typing import List, Optional

from lib.db.db_manager import DatabaseManager
from lib.db.model.file_chunk import FileChunk


class FileChunkService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def add_all(self, chunks: List[FileChunk]):
        with self.db_manager.session_scope() as session:
            new_files = [FileChunk(
                id=uuid.uuid4(),
                name=chunk.name,
                idx=chunk.idx,
                chunk_id=chunk.chunk_id,
                content=chunk.content
            ) for chunk in chunks]

            session.add_all(new_files)

    def find_file_chunks(self, file_name: str) -> List[FileChunk]:
        with self.db_manager.session_scope() as session:
            query = session.query(FileChunk)
            query = query.filter(FileChunk.name == file_name)
            chunks = query.all()

            for chunk in chunks:
                session.expunge(chunk)

            return chunks

    def find_by_id(self, id: uuid) -> Optional[FileChunk]:
        with self.db_manager.session_scope() as session:
            chunk = session.query(FileChunk).get(id)
            if chunk is not None:
                session.expunge(chunk)
        return chunk

    def delete_by_id(self, id: uuid):
        with self.db_manager.session_scope() as session:
            chunk = session.query(FileChunk).get(id)
            if chunk is not None:
                session.delete(chunk)

    def delete_by_file_name(self, file_name: str):
        with self.db_manager.session_scope() as session:
            session.query(FileChunk).filter(FileChunk.name == file_name).delete()
