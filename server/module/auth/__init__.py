from fastapi import APIRouter
from module.auth.endpoint import register, signin, signout, me, token

router = APIRouter(prefix="/auth", tags=["Authentication"])

router.include_router(token.router)
router.include_router(register.router)
router.include_router(signin.router)
router.include_router(signout.router)
router.include_router(me.router)
