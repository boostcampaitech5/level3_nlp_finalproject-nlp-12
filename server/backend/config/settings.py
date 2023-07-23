from pydantic import BaseSettings

class CommonSettings(BaseSettings):
    APP_NAME:   str = "test"
    DEBUG_MODE: bool = False


class ServerSettings(BaseSettings):
    HOST:       str = "0.0.0.0"
    PORT:       int = 0
    REACT_PORT: int = 0


class DatabaseSettings(BaseSettings):
    DB_URL:     str = ''
    DB_NAME:    str = 'test'
    COLLECTION: str = 'user_data'
    USERNAME:   str = ''
    PASSWORD:   str = ''


class ModelSettings(BaseSettings):
    MODEL_NAME      :str = ''
    MODEL_PATH      :str = ''
    EMBEDDING_SIZE  :int = 1536
    COLLECTION      :str = 'message_store'
    OPENAI_API_KEY  :str = ''


class Settings(CommonSettings, ServerSettings, DatabaseSettings, ModelSettings):
    CommonSetting   :CommonSettings     = CommonSettings()
    ServerSetting   :ServerSettings     = ServerSettings()
    DatabaseSetting :DatabaseSettings   = DatabaseSettings()
    ModelSetting    :ModelSettings      = ModelSettings()


settings = Settings()