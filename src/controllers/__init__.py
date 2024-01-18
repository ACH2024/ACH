from .basic_controller import BasicMAC
from .sc_controller import ScMAC

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["sc_mac"] = ScMAC
