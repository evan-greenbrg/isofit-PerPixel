import os
from datetime import datetime, timedelta, timezone
from typing import Annotated
from contextlib import contextmanager
from typing import Optional

import sqlite3
import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel


DB_PATH = os.environ["ISOFIT_USERS_PATH"]


class User(BaseModel):
    username: str
    description: Optional[str] = None
    disabled: Optional[bool] = None


class UserCreate(BaseModel):
    username: str
    password: str
    description: Optional[str] = None


class UserInDB(User):
    hashed_password: str


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_name):
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS {} (
                username TEXT PRIMARY KEY,
                description TEXT,
                hashed_password TEXT NOT NULL,
                disabled INTEGER DEFAULT 0
            )
            """.format(db_name)
        )


def create_user(db_name, user: UserCreate) -> User:
    hashed = PasswordHash.recommended().hash(user.password)
    with get_db() as conn:
        try:
            conn.execute(
                f"""
                INSERT INTO {db_name} 
                (username, description, hashed_password) 
                VALUES (?, ?, ?)
                """,
                (user.username, user.description, hashed),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )
    return get_user(db_name, user.username)


def drop_db(db_name):
    with get_db() as conn:
        conn.execute(
            """
            DROP TABLE {}
            """.format(db_name)
        )


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


def get_users(db_name):
    with get_db() as conn:
        row = conn.execute(
            f"""
            SELECT * 
            FROM {db_name}
            """
        ).fetchone()

    if row:
        return UserInDB(**dict(row))


def delete_user(db_name: str, username: str):
    if get_user(db_name, username):
        with get_db() as conn:
            conn.execute(
                f"""
                DELETE FROM {db_name}
                WHERE username = ?
                """, (username,)
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Username does not exists"
        )


if __name__ == '__main__':
    db_name = 'users'
    init_db('users')
    user = UserCreate(
        username='dev',
        description='Development User',
        password='UserDevelopment'
    )
    create_user(db_name, user)
    print(get_users(db_name))
    # delete_user('users', 'dev')
    # print(get_users(db_name))
    # drop_db('users')


    authenticate_user(db_name, 'dev', 'UserDevelopment')
