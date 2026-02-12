# isofit-PerPixel

### Description

This contains the scripts (and prototype api) to run ISOFIT OE inversions on single pixel-row examples. 

On the ISOFIT-side, this contains a new script to directly build the ISOFIT config, perform the LUT generation for both the presolve and main solve, and return a dictionary with the statevector and OE solution.

On the api-side, this contains a Docker set up for a simple api endpoint and task worker. JSON files with the input data are sent to the endpoint, which kicks off the ISOFIT run. These can be returned as string dictionaries or output as .csv.

Some important considerations:
- I disabled Ray in the Docker environment. I was having issues with the internal multiprocessing.
- I need to hook up a direct, local click CLI for the ISOFIT inversion. The only way to currently run the inversion is via the API call.
- I have a list of things ISOFIT inputs that I haven't hooked up yet (e.g. custom surface files, instrument noise files). Much of the configuration is currently coming from `env.paths`


### How to run

Installing Docker is a pre-requisite. I've only tested locally with the containers running on the same local machine that I send commands from.

1) At the root with `docker-compose.yml`, run: 

    `docker compose up`

    to build the containers.

2) I've provided an example format for the input data in `examples/test_call.json`. Currently, the call contains `RDN`, `LOC`, `OBS`, `wavelength`, `fwhm` lists, and `sensor`, `granule_id`, `rundir` strings, and `delete_all_files`, a bool.

3) To run on the container for the test_call:
```
    curl -X POST http://localhost:8000/process \
    -H "Content-Type: application/json" \
    -d @examples/test_call.json
```

4) Check status of inversion task:
```
curl -X GET http://localhost:8000/result/{task_id}
```

5) Download reslt of inversion task to csv. NOTE: will "wait" for `success` or `failure`
```
curl -o path/to/output.csv http://localhost:8000/download/{task_id}
```
