from typing import List

import streamlit as st

from lib.db.model.user_files import UserFile
from lib.service.user_file_service import UserFileService
from lib.st.cached import db_manager
from lib.utils.session_data import SessionData


class SessionService:
    def __init__(self):
        pass

    @classmethod
    def load_session_data(cls):
        print("Loading session data")

        file_service = UserFileService(db_manager())
        files = file_service.find_user_files()

        SessionService.set_session_data(SessionData(files))

        print("Loading session data completed!")

    @classmethod
    def set_session_data(cls, session_data: SessionData):
        st.session_state["session_data"] = session_data

    @classmethod
    def get_session_data(cls) -> SessionData:
        if "session_data" not in st.session_state:
            SessionService.load_session_data()
        return st.session_state["session_data"]

    @classmethod
    def get_session_files(cls) -> List[UserFile]:
        return SessionService.get_session_data().files

    @classmethod
    def add_to_session_file(cls, file: UserFile) -> SessionData:
        return SessionService.get_session_data().files.append(file)

    @classmethod
    def add_to_session_files(cls, files: List[UserFile]) -> SessionData:
        return SessionService.get_session_data().files.extend(files)

    @classmethod
    def remove_session_file(cls, file: UserFile):
        SessionService.get_session_data().files.remove(file)

    @classmethod
    def is_session_data_set(cls) -> bool:
        return "session_data" in st.session_state and st.session_state["session_data"]
