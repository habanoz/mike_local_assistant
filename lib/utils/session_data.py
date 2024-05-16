from typing import List

from lib.db.model.user_files import UserFile


class SessionData:
    def __init__(self, files: List[UserFile]):
        self._files = files

    @property
    def files(self):
        return self._files
