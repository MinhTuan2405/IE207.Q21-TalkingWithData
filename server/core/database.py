from sqlalchemy import create_engine                 
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .server.env file from the server directory
env_path = Path(__file__).parent.parent / ".server.env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in .server.env file")

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "sslmode": "require",       
        "connect_timeout": 10,       
    },
    pool_pre_ping=True,              
    pool_recycle=3600,               
    pool_size=5,                      
    max_overflow=10                   
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db  
    finally:
        db.close()  
        


if __name__ == "__main__":
    try:
        # Test database connection
        with engine.connect() as connection:
            print("Connection to the database successfully")
            
    except Exception as e:
        print(f"Cannot access the database: {e}")
        print("Please check your DATABASE_URL in .server.env file")