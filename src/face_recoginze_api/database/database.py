from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from face_recoginze_api.settings.database import DatabaseSettings
from typing import AsyncGenerator
from face_recoginze_api.models import models

class Database:
    def __init__(self):
        self.db_config = DatabaseSettings()
        self.db_name = self.db_config.db_name
        username = self.db_config.username
        password = self.db_config.password
        self.hostname = self.db_config.hostname
        self.port = self.db_config.port
        pgsql_url = f"postgresql+asyncpg://{username}:{password}@{self.hostname}:{self.port}/{self.db_name}"
        self.engine = create_async_engine(pgsql_url, echo=True, future=True)
        self.async_session_maker = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)


    async def create_db_and_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(models.SQLModel.metadata.create_all)


    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.async_session_maker() as session:
            yield session  # ✅ Trả về AsyncSession