from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, Session


engine = create_engine("postgresql://postgres:pass@localhost:5432/postgres")


def init_db() -> None:
    from db.model import Base

    Base.metadata.create_all(engine)


def create_session() -> scoped_session[Session]:
    return scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


if __name__ == "__main__":
    init_db()
