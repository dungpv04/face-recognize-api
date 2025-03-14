from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(secrets_dir="/run/secrets", env_file=".env", case_sensitive=True)

    db_name: str = "postgres"
    username: str = "postgres"
    password: str = ""  # Đọc từ Docker Secret nếu có
    hostname: str = "localhost"
    port: int = 5432
