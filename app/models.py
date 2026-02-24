from functools import wraps
import inspect

from pydantic import BaseModel
from typing import Optional, Dict, Any

import numpy as np


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


class TaskList(BaseModel):
    tasks: list


class ResultResponse(BaseModel):
    status: str
    task_id: str
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


class InversionData:
    def __init__(self, data):
        if not isinstance(data, list):
            self.data = [data]
        else:
            self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __bands__(self):
        lens = [len(self.data[i]) for i in range(len(self))]
        if len(set(lens)) <= 1:
            return lens[0]
        else:
            return lens

    def __getitem__(self, index):
        if isinstance(index, list):
            return InversionData([self.data[i] for i in index])

        return self.data[index]

    def __repr__(self):
        return f"InversionData({self.data})"

    def __setitem__(self, index, value):
        self.data[index] = value

    def __array_(self, dtype=None, copy=None):
        return np.array(self.data, dtype=dtype)

    def array(self):
        return np.array(self.data)


class OutputData:
    def __init__(self, statevec, solution):
        self.statevec = statevec
        self.solution = solution

    def __iter__(self):
        return iter(self.solution)

    def __len__(self):
        return len(self.solution)

    def __getitem__(self, index):
        if isinstance(index, list):
            return {
                "statevec": InversionData([self.statevec[i] for i in index]),
                "solution": InversionData([self.solution[i] for i in index]),
            }
        else:
            return {
                "statevec": self.statevec[index],
                "solution": self.solution[index],
            }

    def __setitem__(self, index, value):
        self.statevec[index] = value[0]
        self.solution[index] = value[1]

    def __repr__(self):
        return f"OutputData:\nNumber of rows: {len(self)}"


def enforce_annotations(func):
    hints = inspect.get_annotations(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)

        new_args = {}
        for name, value in bound.arguments.items():
            arg_type = hints.get(name)
            if arg_type and not isinstance(value, arg_type):
                new_args[name] = arg_type(value)
            else:
                new_args[name] = value

        return func(**new_args)
    return wrapper
