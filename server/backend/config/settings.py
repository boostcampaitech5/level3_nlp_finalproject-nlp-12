import os

from dotenv import load_dotenv
from pydantic import BaseSettings, Field

load_dotenv('../../.env')


class CommonSettings(BaseSettings):
    """React 내부 설정."""
    APP_NAME: str = 'test'
    DEBUG_MODE: bool = False


class ServerSettings(BaseSettings):
    """서버 연결을 위한 설정."""
    HOST: str = Field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    PORT: int = Field(default_factory=lambda: os.getenv('PORT', 8000))
    REACT_PORT: int = Field(default_factory=lambda: os.getenv('REACT_PORT', 3000))


class DatabaseSettings(BaseSettings):
    """데이터베이스 설정."""
    DB_URL: str = Field(default_factory=lambda: os.getenv('DB_URL', ''))
    DB_NAME: str = Field(default_factory=lambda: os.getenv('DB_NAME', 'test'))
    COLLECTION: str = Field(default_factory=lambda: os.getenv('COLLECTION', 'user_data'))
    USERNAME: str = Field(default_factory=lambda: os.getenv('USERNAME', ''))
    PASSWORD: str = Field(default_factory=lambda: os.getenv('PASSWORD', ''))


class ModelSettings(BaseSettings):
    """모델과 관련된 키값 등의 설정."""
    MODEL_NAME: str = Field(default_factory=lambda: os.getenv('MODEL_NAME', ''))
    MODEL_PATH: str = Field(default_factory=lambda: os.getenv('MODEL_PATH', ''))
    EMBEDDING_SIZE: int = Field(default_factory=lambda: os.getenv('EMBEDDING_SIZE', 1536))
    COLLECTION: str = Field(default_factory=lambda: os.getenv('MODEL_COLLECTION', 'message_store'))
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))


class Settings(CommonSettings, ServerSettings, DatabaseSettings, ModelSettings):
    """여러 설정을 모아두는 역할."""
    CommonSetting: CommonSettings = CommonSettings()
    ServerSetting: ServerSettings = ServerSettings()
    DatabaseSetting: DatabaseSettings = DatabaseSettings()
    ModelSetting: ModelSettings = ModelSettings()


settings = Settings()
