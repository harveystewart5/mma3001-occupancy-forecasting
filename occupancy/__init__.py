"""Occupancy forecasting for HVAC scheduling at Monash smart buildings.

MMA3001 project (Dataset 2). The package is organised as a pipeline:

1. `occupancy.loaders`  - read the raw CSV exports into tidy tables.
2. `occupancy.audit`    - detect unreliable observations (frozen sensors,
   duplicates, outages) before any modelling.
3. `occupancy.resample` - convert the irregular event stream into an hourly,
   time-weighted headcount series.
4. `occupancy.models`   - baseline and machine-learning forecasters.
5. `occupancy.physics`  - CO2 mass-balance case study (ODE, Euler methods).

Logging (not ``print``) is used for all runtime messages.
"""

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
