import time
import csv
import io
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user
)
from models import (
    InversionInput,
    TaskResponse,
    TaskList,
    ResultResponse,
    Token,
    User,
)
from tasks import run_inversion, get_app_tasks


app = FastAPI(
    title="OE Inversion call",
    description="API to run OE isofit inversion"
)


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(
        form_data.username,
        form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})

    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    return current_user


@app.post(
    "/process",
    response_model=TaskResponse
)
async def submit_task(
    input_data: InversionInput,
    current_user: User = Depends(get_current_active_user)
):
    try:
        data_dict = input_data.model_dump()
        task = run_inversion.delay(data_dict)

        return TaskResponse(
            task_id=task.id,
            status="processing"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/tasks")
async def get_tasks(
    current_user: User = Depends(get_current_active_user)
):

    return TaskList(tasks=get_app_tasks())


@app.get(
    "/result/{task_id}",
    response_model=ResultResponse
)
async def get_result(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    task = AsyncResult(task_id)

    if task.state == "PENDING":
        return ResultResponse(
            status="pending",
            task_id=task_id,
            message="Task is waiting to be processed"
        )
    elif task.state == "PROGRESS":
        return ResultResponse(
            status="processing",
            task_id=task_id,
            message="Task is running"
        )
    elif task.state == "SUCCESS":
        return ResultResponse(
            status="completed",
            task_id=task_id,
            result=task.result
        )
    else:
        return ResultResponse(
            status="failed",
            task_id=task_id,
            result=str(task.info)
        )


@app.delete("/task/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    task = AsyncResult(task_id)
    task.revoce(terminate=True)

    return {"message": f"Task {task_id} cancelled"}


@app.get("/health")
async def health_check(
    current_user: User = Depends(get_current_active_user)
):
    return {"status": "healthy"}


@app.get("/download/{task_id}")
async def download_result(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    task = AsyncResult(task_id)

    if task.state == "failed":
        return ResultResponse(
            status="failed",
            result=str(task.info)
        )

    while task.state != "SUCCESS":
        time.sleep(5)
        task = AsyncResult(task_id)

    result_data = task.result['result']

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write headers
    writer.writerow(['statevec', 'value'])

    # Write data row
    for col, val in zip(
        result_data['statevec'],
        result_data['solution']
    ):
        writer.writerow([col, val])

    csv_string = output.getvalue()
    csv_bytes = io.BytesIO(csv_string.encode('utf-8'))

    return StreamingResponse(
        csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=result_{task_id}.csv"
        }
    )
