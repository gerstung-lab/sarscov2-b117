# type declarations
from jax.numpy import DeviceArray
from typing import Callable, Dict, Optional

Model = Callable[[DeviceArray], DeviceArray]
Guide = Callable[[DeviceArray], None]
