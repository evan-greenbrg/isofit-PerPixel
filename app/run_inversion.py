from collections import Counter
import sys
import time
import logging
import pprint
from datetime import datetime
import itertools
import shutil
from pathlib import Path

import numpy as np
import ray

from isofit.core.common import load_esd
from isofit.core.forward import ForwardModel
from isofit.core.fileio import IO
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
import isofit.utils.template_construction as tmpl
from isofit.configs import Config

from models import InversionData, OutputData, enforce_annotations
from isofit_config import InputConfig, FID


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def oe_inversion(config, fm,
                 rdn_data, loc_data, obs_data,
                 only_converged=True):
    """
    Band order is implicitely hard-coded:
    LOC:
        1 - Longitude
        2 - Latitude
        3 - Elevation
    OBS:
        1 - Path length
        2 - To sensor azimuth
        3 - to sensor zenith
        4 - To sun azimuth
        5 - To sun zenith
        6 - Solar phase
        7 - Slope
        8 - Aspect
        9 - Cos(i)
        10 - UTC time
    """


    geom = Geometry(
        loc=loc_data,
        obs=obs_data,
    )
    iv = Inversion(config, fm)
    try:
        x = iv.invert(rdn_data, geom)
    except ValueError:
        return [-9999. for i in range(fm.nstate)], [-9999. for i in range(fm.nstate)]
    S_hat, K, G = iv.calc_posterior(x[-1], geom, rdn_data)
    posterior_uncertainty = np.sqrt(np.diag(S_hat))

    if only_converged:
        return x[-1, :], posterior_uncertainty
    else:
        return x, posterior_uncertainty


@ray.remote
def ray_oe_inversion(config, fm, rdn_data, loc_data, obs_data):
    return oe_inversion(config, fm, rdn_data, loc_data, obs_data)


def parse_date(sensor, gid):
    dt, _ = tmpl.sensor_name_to_dt(
        sensor,
        FID(sensor)(gid)
    )
    return datetime.strftime(dt, '%Y-%m-%d')


@enforce_annotations
def main(
    rdn_data: InversionData,
    longitude: InversionData,
    latitude: InversionData,
    elevation: InversionData,
    path_length: InversionData,
    sensor_azimuth: InversionData,
    sensor_zenith: InversionData,
    sun_azimuth: InversionData,
    sun_zenith: InversionData,
    solar_phase: InversionData,
    slope: InversionData,
    aspect: InversionData,
    cos_i: InversionData,
    utc_time: InversionData,
    surface_file_paths: InversionData,
    channelized_noise_paths: InversionData,
    parametric_noise_paths: InversionData,
    model_discrepancy_paths: InversionData,
    rdn_factors_paths: InversionData,
    wl: dict,
    fwhm: dict,
    sensor: InversionData,
    gid: InversionData,
    rundir: str,
    n_cores: int = 1,
    delete_all_files: bool = True,
    task_id: str = None,
    inp_batch_keys: list = ['gid'],
    emulator_base=None,
    **kwargs
):
    """
    Not currently hooked up:
      - Specified surface prior file.
        How to pass this in? Could be directly as json
      - Alternative surface category (surface model)
      - Instrument and model uncertainty files
      - Need to double check 6c emulator (multipart transm) hookups.
        This was written with 3c sRTMnet.
      - Run-specific AOD bounds and width of h2o window after presolve
    """

    start_time = time.time()

    if task_id:
        rundir = Path(rundir) / task_id
    else:
        rundir = Path(rundir)

    # Check to make sure we know how to handle the provided batch keys
    valid_batch_keys = ['sensor', 'dt', 'gid']
    for key in inp_batch_keys:
        if key not in valid_batch_keys:
            raise ValueError(
                "Attempting to batch on an invalid key",
                f"Valid batch keys: {valid_batch_keys}"
            )

    # Parse the valid keys
    batch_keys = []
    if 'sensor' in inp_batch_keys:
        batch_keys.append(sensor)

    if 'dt' in inp_batch_keys:
        batch_keys.append(
            list(itertools.starmap(
                parse_date,
                zip(sensor, gid)
            ))
        )

    if 'gid' in inp_batch_keys:
        batch_keys.append(gid)

    # Batch processing if batch keys are provided
    if len(batch_keys):
        iterator = itertools.groupby(
            enumerate(zip(*batch_keys)),
            key=lambda x: x[1]
        )
        batches = {}
        for key, group in iterator:
            batches.setdefault(key, []).extend([i for i, _ in group])

    # Bulk processing if not
    else:
        batches = {tuple(['all']): [i for i in range(len(gid))]}

    # Iterate over batches
    state_names = {}
    states = {}
    uncertainties = {}
    for batch, indexes in batches.items():
        batch_rundir = rundir / '_'.join(batch)
        # Batch input data
        batch_rdn_data = rdn_data[indexes].array()
        batch_longitude = longitude[indexes].array()
        batch_latitude = latitude[indexes].array()
        batch_elevation = elevation[indexes].array()
        batch_path_length = path_length[indexes].array()
        batch_sensor_azimuth = sensor_azimuth[indexes].array()
        batch_sensor_zenith = sensor_zenith[indexes].array()
        batch_sun_azimuth = sun_azimuth[indexes].array()
        batch_sun_zenith = sun_zenith[indexes].array()
        batch_solar_phase = solar_phase[indexes].array()
        batch_slope = slope[indexes].array()
        batch_aspect = aspect[indexes].array()
        batch_cos_i = cos_i[indexes].array()
        batch_utc_time = utc_time[indexes].array()
        batch_sensor = sensor[indexes]
        batch_gid = gid[indexes]
        batch_surface_paths = surface_file_paths[indexes]
        batch_channelized_noise = channelized_noise_paths[indexes]
        batch_parametric_noise = parametric_noise_paths[indexes]
        batch_model_discrepancy = model_discrepancy_paths[indexes]
        batch_rdn_factors = rdn_factors_paths[indexes]

        # Make the LUT config for the entire batch:
        # Sensor and gid: use most common
        # Data arrays: use np.mean
        agg = np.mean
        most_common_sensor = Counter(batch_sensor).most_common(1)[0][0]
        most_common_gid = Counter(batch_gid).most_common(1)[0][0]
        most_common_surface_path = Counter(batch_surface_paths).most_common(1)[0][0]
        most_common_channelized_noise = Counter(batch_channelized_noise).most_common(1)[0][0]
        most_common_parametric_noise = Counter(batch_parametric_noise).most_common(1)[0][0]
        most_common_model_discrepancy = Counter(batch_model_discrepancy).most_common(1)[0][0]
        batch_wl = np.array(wl[most_common_sensor])
        batch_fwhm = np.array(fwhm[most_common_sensor])

        input_config = InputConfig(
            rundir=batch_rundir,
            sensor=most_common_sensor,
            granule_id=most_common_gid,
            elevation_data=agg(batch_elevation),
            sensor_zenith_data=agg(batch_sensor_zenith),
            sun_zenith_data=agg(batch_sun_zenith),
            sensor_azimuth_data=agg(batch_sensor_azimuth),
            sun_azimuth_data=agg(batch_sun_azimuth),
            path_length=agg(batch_path_length),
            utc_time=agg(batch_utc_time),
            wl=batch_wl,
            fwhm=batch_fwhm,
            surface_json_path=most_common_surface_path,
            n_cores=n_cores,
            channelized_uncertainty_file=most_common_channelized_noise,
            parametric_noise_file=most_common_parametric_noise,
            model_discrepancy_file=most_common_model_discrepancy,
            emulator_base=emulator_base
        )

        # Assemble expected ISOFIT input format
        batch_loc_data = np.zeros((batch_rdn_data.shape[0], 3))
        batch_obs_data = np.zeros((batch_rdn_data.shape[0], 10))
        for i in range(len(batch_rdn_data)):
            batch_loc_data[i, :] = np.stack([
                batch_longitude[i],
                batch_latitude[i],
                batch_elevation[i],
            ], axis=0)

            batch_obs_data[i, :] = np.stack([
                batch_path_length[i],
                batch_sensor_azimuth[i],
                batch_sensor_zenith[i],
                batch_sun_azimuth[i],
                batch_sun_zenith[i],
                batch_solar_phase[i],
                batch_slope[i],
                batch_aspect[i],
                batch_cos_i[i],
                batch_utc_time[i],
            ], axis=0)

        # Assume if rdn_factors exist, they are uniform across batch
        most_common_rdn_factors_file = Counter(batch_rdn_factors).most_common(1)[0][0]
        if most_common_rdn_factors_file:
            rdn_factors = np.loadtxt(most_common_rdn_factors_file) 
            assert len(rdn_factors) == len(batch_wl)
        else:
            rdn_factors = np.ones(len(batch_wl))

        batch_rdn_data *= rdn_factors[None, :]

        # PRESOLVE
        modtran_template_path = (
            input_config.config_root
            / (input_config.fid + "_h2o_tpl.json")
        )
        lut_directory = input_config.rundir / "lut_h2o"
        lut_directory.mkdir(exist_ok=True)

        presolve_config = input_config.build(
            str(modtran_template_path),
            str(lut_directory),
            presolve=True,
            retrieve_co2=False,
            h2o_min=kwargs.get("h2o_min", 0.2),
            h2o_max=kwargs.get("h2o_max", 5.6),
            h2o_spacing=0.64,
        )
        dict_str = pprint.pformat(presolve_config, indent=1)
        logging.debug(dict_str)

        presolve_config = Config(presolve_config)
        logging.info(f"isofit presolve n_cores: {presolve_config.implementation.n_cores}")

        # Instantiate FM (will build lut). Have to rely on isofit ray?
        logging.info(f"Building presolve LUT")
        fm = ForwardModel(presolve_config)
        statevec = fm.statevec

        # Parallelize pixel loop via Ray shared memory
        print("Running presolve")
        fm_ref = ray.put(fm)
        futures = [
            ray_oe_inversion.remote(
                presolve_config,
                fm_ref,
                batch_rdn_data[i],
                batch_loc_data[i],
                batch_obs_data[i]
            )
            for i in range(len(batch_rdn_data))
        ]
        batch_results = np.array(ray.get(futures))
        batch_state = batch_results[:, 0, :]
        batch_unc = batch_results[:, 1, :]

        # Handle the presolve results
        h2o_idx = [i for i, val in enumerate(statevec) if val == 'H2OSTR'][0]
        h2o_est = batch_state[:, h2o_idx]
        p05 = np.percentile(h2o_est[h2o_est > kwargs.get("h2o_min", 0.2)], 2)
        p95 = np.percentile(h2o_est[h2o_est < kwargs.get("h2o_max", 5.6)], 98)

        margin = (p95 - p05) * 0.5
        h2o_spacing = 0.25
        h2o_min = max(min(p05 - margin, p05 - h2o_spacing), 0)
        h2o_max = min(max(p95 + margin, p95 + h2o_spacing), 6)

        # Main solve
        modtran_template_path = (
            input_config.config_root
            / (input_config.fid + "_modtran_tpl.json")
        )
        lut_directory = input_config.rundir / "lut_full"
        lut_directory.mkdir(exist_ok=True)

        main_config = input_config.build(
            str(modtran_template_path),
            str(lut_directory),
            presolve=False,
            retrieve_co2=False,
            h2o_min=kwargs.get("h2o_min", 0.2),
            h2o_max=kwargs.get("h2o_max", 5.6),
            h2o_spacing=h2o_spacing,
            aerosol_min=kwargs.get("aerosol_min", 0.05),
            aerosol_max=kwargs.get("aerosol_max", 0.5),
            **kwargs
        )
        dict_str = pprint.pformat(main_config, indent=1)
        logging.debug(dict_str)

        main_config = Config(main_config)
        print(main_config.forward_model.atmosphere.emulator_file)
        logging.info(f"isofit main solve n_cores: {main_config.implementation.n_cores}")
        logging.info(f"Building main solve LUT — n_cores={n_cores}")
        fm = ForwardModel(main_config)
        statevec = fm.statevec

        print("Running main solve")
        fm_ref = ray.put(fm)
        futures = [
            ray_oe_inversion.remote(
                main_config, 
                fm_ref,
                batch_rdn_data[i],
                batch_loc_data[i],
                batch_obs_data[i]
            )
            for i in range(len(batch_rdn_data))
        ]
        results = ray.get(futures)
        batch_results = np.array(ray.get(futures))
        batch_state = batch_results[:, 0, :]
        batch_unc = batch_results[:, 1, :]

        state_names[batch] = statevec
        states[batch] = batch_state
        uncertainties[batch] = batch_unc

        if delete_all_files:
            shutil.rmtree(input_config.rundir)

    output = OutputData(
        InversionData([[] for i in range(len(rdn_data))]),
        InversionData([[] for i in range(len(rdn_data))]),
        InversionData([[] for i in range(len(rdn_data))])
    )
    for key, idx in batches.items():
        for state, unc, i in zip(states[key], uncertainties[key], idx):
            output[i] = (
                state_names[key],
                [float(s) for s in state],
                [float(v) for v in unc],
            )

    end_time = time.time()
    logging.info(f"Batch completed in {round(end_time - start_time, 2)} seconds")

    return {
        'statevec': list(output.statevec),
        'solution': list(output.solution),
        'uncertainty': list(output.uncertainty),
        'runtime_seconds': round(end_time - start_time, 2)
    }
