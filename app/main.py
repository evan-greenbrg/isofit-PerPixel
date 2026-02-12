import time
import csv
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from models import InversionInput, TaskResponse, ResultResponse
from tasks import run_inversion


app = FastAPI(
    title="OE Inversion call",
    description="API to run OE isofit inversion"
)


@app.post(
    "/process",
    response_model=TaskResponse,
    status_code=202
)
async def submit_task(input_data: InversionInput):
    try:
        data_dict = input_data.model_dump()
        task = run_inversion.delay(data_dict)

        return TaskResponse(
            task_id=task.id,
            status="processing"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/result/{task_id}",
    response_model=ResultResponse
)
async def get_result(task_id: str):
    task = AsyncResult(task_id)

    if task.state == "PENDING":
        return ResultResponse(
            status="pending",
            message="Task is waiting to be processed"
        )
    elif task.state == "PROGRESS":
        return ResultResponse(
            status="processing",
            message="Task is running"
        )
    elif task.state == "SUCCESS":
        return ResultResponse(
            status="completed",
            result=task.result
        )
    else:
        return ResultResponse(
            status="failed",
            result=str(task.info)
        )


@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    task = AsyncResult(task_id)
    task.revoce(terminate=True)

    return {"message": f"Task {task_id} cancelled"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/download/{task_id}")
async def download_result(task_id: str):
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
