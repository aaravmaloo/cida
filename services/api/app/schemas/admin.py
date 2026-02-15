from datetime import datetime

from pydantic import BaseModel


class AdminLoginRequest(BaseModel):
    passkey: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

