import uuid
from typing import List, Optional

from lib.db.db_manager import DatabaseManager
from lib.db.model.user_files import UserFile


class UserFileService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save(self, user_file: UserFile):
        with self.db_manager.session_scope() as session:
            new_file = UserFile(
                id=user_file.id,
                name=user_file.name,
                content=user_file.content
            )

            session.add(new_file)

            file = UserFile(id=new_file.id, name=new_file.name, content=new_file.content)
        return file

    def add(self, name: str, content: str):
        with self.db_manager.session_scope() as session:
            new_file = UserFile(
                id=uuid.uuid4(),
                name=name,
                content=content
            )

            session.add(new_file)

            user = UserFile(id=new_file.id, name=name)

        return user

    def add_all(self, files: List[UserFile]):
        with self.db_manager.session_scope() as session:
            new_files = [UserFile(
                id=uuid.uuid4(),
                name=file.name,
                content=file.content
            ) for file in files]

            session.add_all(new_files)

    def find_user_files(self) -> List[UserFile]:
        with self.db_manager.session_scope() as session:
            query = session.query(UserFile)
            files = query.all()

            for file in files:
                session.expunge(file)

            return files

    def find_by_id(self, id: uuid) -> Optional[UserFile]:
        with self.db_manager.session_scope() as session:
            user_file = session.query(UserFile).get(id)
            if user_file is not None:
                session.expunge(user_file)
        return user_file

    def delete_by_id(self, id: uuid):
        with self.db_manager.session_scope() as session:
            user = session.query(UserFile).get(id)
            if user is not None:
                session.delete(user)
