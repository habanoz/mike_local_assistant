import uuid

from sqlalchemy.orm import Query

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

    def add_utterance(self, chat_id: uuid, type: str, message: str, files=None, debug=None):
        with self.db_manager.session_scope() as session:
            new_utterance = Utterance(
                id=uuid.uuid4(),
                chat_id=chat_id,
                type=type,
                message=message,
                files=files if files else None,
                debug=debug if debug else None
            )

            session.add(new_utterance)

    def update_utterance_alternatives(self, utterance):
        with self.db_manager.session_scope() as session:
            session.query(Utterance).filter(Utterance.id == utterance.id).update(
                {"alternatives": utterance.alternatives})

    def fetch_by_chat_id(self, chat_id: uuid) -> list[Utterance]:
        with self.db_manager.session_scope() as session:
            query: Query = session.query(Utterance)
            query = query.filter(Utterance.chat_id == chat_id)

            utterances = query.all()

            for utterance in utterances:
                session.expunge(utterance)

            return utterances

    def fetch_chat_by_id(self, chat_id: uuid) -> ChatSession:
        with self.db_manager.session_scope() as session:
            chat_session = session.query(ChatSession).get(chat_id)
            if chat_session is not None:
                session.expunge(chat_session)
        return chat_session

    def fetch_recent(self) -> list[Utterance]:
        with self.db_manager.session_scope() as session:
            query: Query = session.query(Utterance)
            query = query.order_by(Utterance.created.desc()).limit(10)

            utterances = query.all()

            for utterance in utterances:
                session.expunge(utterance)

            return utterances

    def fetch_recent_chats(self) -> list[ChatSession]:
        with self.db_manager.session_scope() as session:
            query: Query = session.query(ChatSession)
            query = query.order_by(ChatSession.created.desc()).limit(10)

            chats = query.all()

            for chat in chats:
                session.expunge(chat)

            return chats
