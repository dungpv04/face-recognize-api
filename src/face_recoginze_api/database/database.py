from sqlmodel import Session, create_engine
from database import tables
from contextlib import contextmanager
class Database:
    def __init__(self):
        self.db_name = "face_recognize"
        username = "postgres"
        password = "Dung200409"
        self.hostname = "localhost"
        self.port = "5432"
        pgsql_url = f"postgresql://{username}:{password}@{self.hostname}:{self.port}/{self.db_name}"

        self.engine = create_engine(pgsql_url)


    def create_db_and_tables(self):
        tables.SQLModel.metadata.create_all(self.engine)
        print(f"connected to database: {self.db_name} on {self.hostname}:{self.port}")


    @contextmanager
    def get_session(self):
        try:
            with Session(self.engine) as session:
                yield session
        except Exception as e:
            print(f"Failed to connect to DB error: {e}")
            session.rollback()
            raise
        finally:
            session.close()