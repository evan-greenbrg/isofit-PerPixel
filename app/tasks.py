import traceback
import numpy as np
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
