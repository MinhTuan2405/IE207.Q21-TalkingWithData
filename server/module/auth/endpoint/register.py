from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from core.database import get_db
from module.auth.schema.auth_schema import UserRegister, TokenResponse
from module.auth.service import auth_service

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    return auth_service.register(db, user_data)