from contextlib import contextmanager
import os
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from pwdlib import PasswordHash
from fastapi.security import OAuth2PasswordBearer
import sqlite3
import jwt
from jwt.exceptions import InvalidTokenError

from models import (
    TokenData,
    User,
    UserInDB,
)


ALGORITHM = "HS256"
DB_PATH = os.environ["ISOFIT_USERS_PATH"]
DB_NAME = 'users'
SECRET_KEY = os.environ["ISOFIT_SECRET_KEY"]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    user = get_user(DB_NAME, username)

    if not PasswordHash.recommended().verify(
        password,
        user.hashed_password
    ):
        return False

    return user


def get_user(db_name: str, username: str) -> Optional[UserInDB]:
    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT *
            FROM {db_name}
            WHERE username = ?
            """, (username,)
        ).fetchone()

    if row:
        return UserInDB(**dict(row))


def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except InvalidTokenError:
        raise credentials_exception

    user = get_user(DB_NAME, username=token_data.username)

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user
