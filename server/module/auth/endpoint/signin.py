from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from module.auth.schema.auth_schema import UserSignIn, TokenResponse
from module.auth.service import auth_service

router = APIRouter()


@router.post("/signin", response_model=TokenResponse)
def signin(signin_data: UserSignIn, db: Session = Depends(get_db)):
    return auth_service.signin(db, signin_data)