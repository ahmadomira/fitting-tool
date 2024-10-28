# Project Structure

## Directories

- **notebooks/**: the Jupyter notebooks for fitting
- **data/**: sample data files for testing


## Files
- **`main_interface.py`**: the main script for running the user interface
- **`IDA_fitting_replica_dye_alone.py`**: script for fitting the IDA model to dye only data

# Building PEX Files
To build a PEX file, run the following command:
```bash
python -m pex -e fitting_tool -o aux.pex -r requirements.txt -D . --scie eager
```
