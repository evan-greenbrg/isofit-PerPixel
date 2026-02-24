import traceback
import json
import numpy as np
from kombu import Connection
from kombu.simple import SimpleQueue

from celery_config import celery_app as app
from run_inversion import main as inversion



@app.task(bind=True)
def run_inversion(self, input_data):
    try:
        self.update_state(
            state="PROGRESS",
            meta={"status": "processing", "progress": 0}
        )

        result = inversion(
            np.array(input_data["rdn_data"]),
            np.array(input_data["loc_data"]),
            np.array(input_data["obs_data"]),
            np.array(input_data["wl"]),
            np.array(input_data["fwhm"]),
            input_data["sensor"],
            input_data["gid"],
            input_data["rundir"],
            input_data["delete_all_files"],
            self.request.id
        )

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": str(traceback.format_exc()),
            "error_type": type(e).__name__
        }


@app.task(bind=True)
def get_app_tasks(self):
    result = []
    # Get unallocated jobs
    with self.app.connection() as conn:
        client = conn.channel().client

        tasks = client.lrange('celery', 0, -1)
        for t in tasks:
            message = json.loads(t)
            worker_task = {
                'id': message['headers']['id'],
                'name': message['headers'].get('name', 'none'),
                'status': "Pending"
            }
            result.append(worker_task)

        # Completed/active tasks
        keys = client.keys('celery-task-meta-*')
        for key in keys:
            message = json.loads(client.get(key))
            worker_task = {
                'id': message['task_id'],
                'name': message.get('name', 'none'),
                'status': "Active/Completed"
            }
            result.append(worker_task)

    return result
