from sqlalchemy import Column, Integer, String, Date, Text, LargeBinary
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Patients(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    lastname = Column(String(64), nullable=False)
    birth_date = Column(Date, nullable=False)

    condition = Column(String(128), nullable=True)
    prescriptions = Column(ARRAY(Text), nullable=True)
    diagnosis = Column(Text, nullable=True)
    original_image = Column(LargeBinary, nullable=True)
    predicted_image = Column(LargeBinary, nullable=True)
