from fastapi import APIRouter, Depends
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User
from module.auth.schema.auth_schema import UserResponse

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user_oauth2)):
    return UserResponse.model_validate(current_user)
