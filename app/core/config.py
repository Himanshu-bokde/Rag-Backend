from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY:str
    REDIS_URL: str = "redis://localhost:6379/0"
    UPLOAD_DIR: str = "storage/uploads"
    MAX_FILE_SIZE_MB: int = 20

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()