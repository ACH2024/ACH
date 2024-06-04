from .basic_controller import BasicMAC
from .cate_broadcast_comm_controller import CateBCommMAC
from .cate_broadcast_comm_controller_full import CateBCommFMAC
from .cate_broadcast_comm_controller_not_IB import CateBCommNIBMAC
from .tar_comm_controller import TarCommMAC
from .cate_pruned_broadcast_comm_controller import CatePBCommMAC
from .ach_controller import AchMAC
from .masia_controller import MASIAMAC
from .maic_controller import MAICMAC

REGISTRY = {"basic_mac": BasicMAC,
            "cate_broadcast_comm_mac": CateBCommMAC,
            "cate_broadcast_comm_mac_full": CateBCommFMAC,
            "cate_broadcast_comm_mac_not_IB": CateBCommNIBMAC,
            "tar_comm_mac": TarCommMAC,
            "cate_pruned_broadcast_comm_mac": CatePBCommMAC,
            "ach_mac": AchMAC,
            "masia_mac": MASIAMAC,
            "maic_mac": MAICMAC}
