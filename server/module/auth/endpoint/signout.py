from fastapi import APIRouter, Depends
from core.dependencies import get_current_user_oauth2
from module.auth.model.user import User

router = APIRouter()


@router.post("/signout")
def signout(current_user: User = Depends(get_current_user_oauth2)):
    return {"message": "Successfully signed out"}