from pydantic import BaseSettings

class CommonSettings(BaseSettings):
    APP_NAME: str = "test"
    DEBUG_MODE: bool = False


class ServerSettings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 0
    REACT_PORT: int = 0


class DatabaseSettings(BaseSettings):
    DB_URL: str = ''
    DB_NAME: str = ''
    COLLECTION: str = ''
    USERNAME: str = ''
    PASSWORD: str = ''


class Settings(CommonSettings, ServerSettings, DatabaseSettings):
    pass


settings = Settings()