import json
from os.path import split
import os
from pathlib import Path

import numpy as np

from isofit.core import units
from isofit.data import env
import isofit.utils.template_construction as tmpl
from isofit.utils.surface_model import surface_model

from models import enforce_annotations

INVERSION_WINDOWS = [[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]]


class InputConfig:
    def __init__(
        self,
        rundir: str,
        sensor: str,
        granule_id: str,
        elevation_data: np.array,
        sensor_zenith_data: np.array,
        sun_zenith_data: np.array,
        sensor_azimuth_data: np.array,
        sun_azimuth_data: np.array,
        path_length: np.array,
        utc_time: np.array,
        wl: np.array,
        fwhm: np.array,
        surface_json_path: str = None,
        engine_name: str = 'sRTMnet',
        aerosol_tpl_path: str = None,
        earth_sun_distance_path: str = None,
        irradiance_file: str = None,
        sixs_path: str = None,
        modtran_path: str = None,
        emulator_base: str = None,
        ray_temp_dir: str = '/tmp/ray',
        ray_address: str = None,
        atmosphere_type="ATM_MIDLAT_SUMMER",
        surface_category="multicomponent_surface",
        terrain_style: str = "flat",
        cos_i_min: float = 0.3,
        n_cores: int = 1,
    ):
        """
        Generic config class to make config.
        """
        self.rundir = rundir
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.sensor = sensor
        self.granule_id = granule_id
        self.fid = FID(sensor)(granule_id)

        self.elevation_data = elevation_data
        self.sensor_zenith_data = sensor_zenith_data
        self.sun_zenith_data = sun_zenith_data
        self.sensor_azimuth_data = sensor_azimuth_data
        self.sun_azimuth_data = sun_azimuth_data

        self.path_length = path_length
        self.utc_time = utc_time

        self.wl = wl
        self.fwhm = fwhm

        self.config_root = self.rundir / "config"
        self.config_root.mkdir(exist_ok=True)

        self.surface_json_path = (
            surface_json_path
            or str(env.path("surface", "surface_20240103_avirii.json"))
        )
        self.data_root = self.rundir / "data"
        self.data_root.mkdir(exist_ok=True)
        self.surface_mat_path = str(
            self.data_root
            / Path(self.surface_json_path).with_suffix(".mat").name
        )
        self.wavelength_path = self.data_root / "wavelengths.txt"

        self.aerosol_tpl_path = (
            aerosol_tpl_path
            or str(env.path("data", "aerosol_template.json"))
        )
        self.earth_sun_distance_path = (
            earth_sun_distance_path
            or str(env.path("data", "earth_sun_distance.txt"))
        )

        irr_path = [
            "examples",
            "20151026_SantaMonica",
            "data",
            "prism_optimized_irr.dat",
        ]
        self.irradiance_file = (
            irradiance_file
            or str(env.path(*irr_path))
        )

        if engine_name == 'sRTMnet':
            self.sixs_path = (
                sixs_path
                or os.getenv("SIXS_DIR", env.sixs)
            )
            self.emulator_base = (
                emulator_base
                or str(env.path("srtmnet", key="srtmnet.file"))
            )
            self.modtran_path = None
        elif engine_name == 'modtran':
            self.modtran_path = (
                modtran_path
                or os.getenv("MODTRAN_DIR", env.modtran)
            )
            self.emulator_base = None
            self.sixs_path = None
            self.multipart_transmittance = True
        else:
            raise ValueError(
                "Only sRTMnet and Modtran are supported RTEs at this time"
            )

        if self.emulator_base is not None:
            if self.emulator_base.endswith(".jld2"):
                self.multipart_transmittance = False

            elif self.emulator_base.endswith(".h5"):
                self.multipart_transmittance = False

            elif self.emulator_base.endswith(".6c"):
                self.multipart_transmittance = True

        self.ray_temp_dir = ray_temp_dir
        self.ray_address = ray_address
        self.atmosphere_type = atmosphere_type
        self.surface_category = surface_category
        self.inversion_windows = INVERSION_WINDOWS

        # Noise files not hooked up yet
        self.input_channelized_uncertainty_path = None
        self.channelized_uncertainty_working_path = None
        self.eof_path = None
        self.eof_working_path = None
        self.noise_path = None
        self.uncorrelated_radiometric_uncertainty = None
        self.dn_uncertainty_file = None
        self.input_model_discrepancy_path = None

        self.terrain_style = terrain_style
        self.cos_i_min = cos_i_min

        self.n_cores = n_cores

    def __repr__(self):
        return json.dumps(
            {self.__class__.__name__: vars(self)},
            indent=4,
            default=str
        )

    def make_lut_grids(self, elevation_data, sensor_zenith_data,
                       sun_zenith_data, sensor_azimuth_data,
                       sun_azimuth_data, aerosol_min, aerosol_max,
                       h2o_min, h2o_max, h2o_spacing,
                       pressure_elevation):

        # Hard coded h2o spacing for now
        lut_params = tmpl.LUTConfig(
            emulator=self.emulator_base,
            h2o_range=[h2o_min, h2o_max],
            h2o_spacing=h2o_spacing,
            aerosol_0_range=[aerosol_min, aerosol_max],
            aerosol_1_range=[aerosol_min, aerosol_max],
            aerosol_2_range=[aerosol_min, aerosol_max],
            aot550_range=[aerosol_min, aerosol_max],
        )

        h2o_lut_grid = lut_params.get_grid(
            lut_params.h2o_range[0],
            lut_params.h2o_range[1],
            lut_params.h2o_spacing,
            lut_params.h2o_spacing_min,
        )

        to_sensor_zenith_lut_grid = lut_params.get_grid_with_data(
            sensor_zenith_data,
            lut_params.to_sensor_zenith_spacing,
            lut_params.to_sensor_zenith_spacing_min,
        )

        to_sun_zenith_lut_grid = lut_params.get_grid_with_data(
            sun_zenith_data,
            lut_params.to_sun_zenith_spacing,
            lut_params.to_sun_zenith_spacing_min,
        )

        delta_phi = np.abs(sun_azimuth_data - sensor_azimuth_data)
        relative_azimuth = np.minimum(
            delta_phi, 360 - delta_phi
        )
        relative_azimuth_lut_grid = lut_params.get_grid_with_data(
            relative_azimuth,
            lut_params.relative_azimuth_spacing,
            lut_params.relative_azimuth_spacing_min,
        )

        elevation_lut_grid = lut_params.get_grid(
            units.m_to_km(np.min(elevation_data)),
            units.m_to_km(np.max(elevation_data)),
            lut_params.elevation_spacing,
            lut_params.elevation_spacing_min,
        )

        # Currently no support for custom climatology configs
        (
            aerosol_state_vector,
            aerosol_lut_grid,
            aerosol_model_path
        ) = tmpl.load_climatology(
            config_path=None,
            latitude=None,
            longitude=None,
            acquisition_datetime=None,
            lut_params=lut_params,
        )

        return (
            h2o_lut_grid,
            to_sensor_zenith_lut_grid,
            to_sun_zenith_lut_grid,
            relative_azimuth_lut_grid,
            elevation_lut_grid,
            aerosol_state_vector,
            aerosol_lut_grid,
            aerosol_model_path,
            lut_params.co2_range
        )

    def wavelengths(self):
        np.savetxt(
            self.wavelength_path,
            np.array([
                units.nm_to_micron(self.wl),
                units.nm_to_micron(self.fwhm)
            ]).T
        )
        return self.wavelength_path

    def surface(self):
        surface_model(
            config_path=self.surface_json_path,
            wavelength_path=self.wavelengths(),
            output_path=self.surface_mat_path,
        )

        return {self.surface_category: self.surface_mat_path}

    def build(
        self,
        modtran_template_path: str,
        lut_directory: str,
        h2o_min: float = 0.2,
        h2o_max: float = 6.0,
        h2o_spacing: float = 0.25,
        aerosol_min: float = 0,
        aerosol_max: float = 1.0,
        presolve: bool = True,
        pressure_elevation: bool = False,
        retrieve_co2: bool = False,
    ):
        # Metadata from loc
        elevation_km = max(
            units.m_to_km(self.elevation_data),
            0
        )

        # Metadata from obs
        path_km = units.m_to_km(self.path_length)
        to_sensor_azimuth = self.sensor_azimuth_data
        to_sensor_zenith = self.sensor_zenith_data
        to_sun_azimuth = self.sun_azimuth_data
        to_sun_zenith = self.sun_zenith_data

        delta_phi = np.abs(to_sun_azimuth - to_sensor_azimuth)
        relative_azimuth = np.minimum(delta_phi, 360 - delta_phi)
        altitude_km = (
            elevation_km
            + (
                np.cos(np.deg2rad(to_sensor_zenith))
                * path_km
            )
        )

        # Date stuff
        dt, sensor_inversion_windows = tmpl.sensor_name_to_dt(
            self.sensor,
            self.fid
        )
        if sensor_inversion_windows:
            self.inversion_windows = sensor_inversion_windows

        # Don't think I need to day increment here
        dayofyear = dt.timetuple().tm_yday
        h_m_s = [np.floor(self.utc_time)]
        h_m_s.append(np.floor((self.utc_time - h_m_s[-1]) * 60))
        h_m_s.append(np.floor(
            (self.utc_time - h_m_s[-2] - h_m_s[-1] / 60.0)
            * 3600
        ))
        if h_m_s[0] != dt.hour and h_m_s[0] >= 24:
            h_m_s[0] = dt.hour
        if h_m_s[1] != dt.minute and h_m_s[1] >= 60:
            h_m_s[1] = dt.minute

        gmtime = float(h_m_s[0] + h_m_s[1] / 60.0)

        # Write modtran template
        tmpl.write_modtran_template(
            atmosphere_type=self.atmosphere_type,
            fid=self.fid,
            altitude_km=altitude_km,
            dayofyear=dayofyear,
            to_sensor_azimuth=to_sensor_azimuth,
            to_sensor_zenith=to_sensor_zenith,
            to_sun_zenith=to_sun_zenith,
            relative_azimuth=relative_azimuth,
            gmtime=gmtime,
            elevation_km=elevation_km,
            output_file=modtran_template_path,
            ihaze_type="AER_NONE" if presolve else "AER_RURAL",
        )

        # Deal with the surface model
        surface_config = tmpl.make_surface_config(
            surface_working_paths=self.surface(),
            surface_category=self.surface_category
        )
        instrument_config = tmpl.make_instrument_config(
            self.wavelength_path,
            self.input_channelized_uncertainty_path,
            self.channelized_uncertainty_working_path,
            self.eof_path,
            self.eof_working_path,
            self.noise_path,
            self.uncorrelated_radiometric_uncertainty,
            self.dn_uncertainty_file,
        )
        implementation_config = tmpl.make_implementation_config(
            ray_temp_dir=self.ray_temp_dir,
            inversion_windows=self.inversion_windows,
            n_cores=self.n_cores
        )

        (
            h2o_lut_grid,
            to_sensor_zenith_lut_grid,
            to_sun_zenith_lut_grid,
            relative_azimuth_lut_grid,
            elevation_lut_grid,
            aerosol_state_vector,
            aerosol_lut_grid,
            aerosol_model_path,
            co2_range
        ) = self.make_lut_grids(
            self.elevation_data,
            self.sensor_zenith_data,
            self.sun_zenith_data,
            self.sensor_azimuth_data,
            self.sun_azimuth_data,
            aerosol_min,
            aerosol_max,
            h2o_min=h2o_min,
            h2o_max=h2o_max,
            h2o_spacing=h2o_spacing,
            pressure_elevation=pressure_elevation
        )

        # Presolve overrides
        aerosol_lut_grid = (
            None if presolve
            else aerosol_lut_grid
        )
        aerosol_model_file = (
            None if presolve
            else aerosol_model_path
        )
        aerosol_state_vector = (
            None if presolve
            else aerosol_state_vector
        )
        co2_lut_grid = (
            co2_range if retrieve_co2
            else None
        )
        elevation_lut_grid = (
            None if presolve
            else elevation_lut_grid
        )
        relative_azimuth_lut_grid = (
            None if presolve
            else relative_azimuth_lut_grid
        )
        to_sensor_zenith_lut_grid = (
            None if presolve
            else to_sensor_zenith_lut_grid
        )
        to_sun_zenith_lut_grid = (
            None if presolve
            else to_sun_zenith_lut_grid
        )

        rt_config = tmpl.make_rt_config(
            lut_directory=lut_directory,
            modtran_template_path=modtran_template_path,
            aerosol_tpl_path=self.aerosol_tpl_path,
            earth_sun_distance_path=self.earth_sun_distance_path,
            irradiance_file=self.irradiance_file,
            sixs_path=self.sixs_path,
            modtran_path=self.modtran_path,
            h2o_lut_grid=h2o_lut_grid,
            aerosol_lut_grid=aerosol_lut_grid,
            aerosol_model_file=aerosol_model_file,
            aerosol_state_vector=aerosol_state_vector,
            co2_lut_grid=co2_lut_grid,
            elevation_lut_grid=elevation_lut_grid,
            emulator_base=self.emulator_base,
            multipart_transmittance=self.multipart_transmittance,
            presolve=presolve,
            pressure_elevation=pressure_elevation,
            retrieve_co2=retrieve_co2,
            relative_azimuth_lut_grid=relative_azimuth_lut_grid,
            to_sensor_zenith_lut_grid=to_sensor_zenith_lut_grid,
            to_sun_zenith_lut_grid=to_sun_zenith_lut_grid,
            terrain_style=self.terrain_style,
            cos_i_min=self.cos_i_min,
        )

        return {
            "forward_model": {
                "instrument": instrument_config,
                "radiative_transfer": rt_config,
                "surface": surface_config,
                "model_discrepancy_file": self.input_model_discrepancy_path
            },
            "implementation": implementation_config
        }


@enforce_annotations
def FID(sensor: str):
    calls = {
        'ang': lambda path: split(path)[-1][:18],
        'av3': lambda path: split(path)[-1][:18],
        'av5': lambda path: split(path)[-1][:18],
        'avcl': lambda path: split(path)[-1][:16],
        'emit': lambda path: split(path)[-1][:19],
        'enmap': lambda path: split(path)[-1].split("_")[5],
        'hyp': lambda path: split(path)[-1][:22],
        'neon': lambda path: split(path)[-1][:21],
        'prism': lambda path: split(path)[-1][:18],
        'prisma': lambda path: path.split("/")[-1].split("_")[1],
        'gao': lambda path: split(path)[-1][:23],
        'oci': lambda path: split(path)[-1][:24],
        'tanager': lambda path: split(path)[-1][:23],
        'NA-': lambda path: os.path.splitext(os.path.basename(path))[0],
    }
    return calls[sensor.lower()]
