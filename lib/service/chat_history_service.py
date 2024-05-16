import uuid

from lib.db.db_manager import DatabaseManager
from lib.db.model.chat_session import ChatSession
from lib.db.model.utterance import Utterance


class ChatHistoryService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def add_chat(self, chat_id: uuid):
        with self.db_manager.session_scope() as session:
            new_chat = ChatSession(
                id=chat_id
            )

            session.add(new_chat)

    def add_utterance(self, chat_id: uuid, type: str, message: str):
        with self.db_manager.session_scope() as session:
            new_utterance = Utterance(
                id=uuid.uuid4(),
                chat_id=chat_id,
                type=type,
                message=message
            )

            session.add(new_utterance)
