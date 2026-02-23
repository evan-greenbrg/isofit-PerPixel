from pydantic import BaseModel
from typing import Optional, Dict, Any


class InversionInput(BaseModel):
    rdn_data: list[float]
    loc_data: list[float]
    obs_data: list[float]
    wl: list[float]
    fwhm: list[float]
    sensor: str
    gid: str
    rundir: str
    delete_all_files: bool = False


class TaskResponse(BaseModel):
    task_id: str
    status: str


class ResultResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class TokenData(BaseModel):
    username: str | None = None


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    description: str = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str
