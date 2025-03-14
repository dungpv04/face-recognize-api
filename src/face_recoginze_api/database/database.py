from sqlmodel import Session, create_engine
from database import tables
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

class Database:
    def __init__(self):
        self.db_name = "postgres"
        username = "postgres"
        password = "Dung200409"
        self.hostname = "localhost"
        self.port = "5433"
        pgsql_url = f"postgresql+asyncpg://{username}:{password}@{self.hostname}:{self.port}/{self.db_name}"
        #self.engine = create_engine(pgsql_url)
        self.engine = create_async_engine(pgsql_url, echo=True, future=True)
        self.async_session_maker = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)


    async def create_db_and_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(tables.SQLModel.metadata.create_all)


    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        async with self.async_session_maker() as session:
            yield session  # ✅ Trả về AsyncSession