"""
Comprehensive DPDK Python Bindings for Djinn Zero-Copy Transport

This module provides ctypes-based bindings to DPDK libraries for:
- EAL (Environment Abstraction Layer) initialization
- Mempool creation and management
- Ethernet device configuration and control
- Packet buffer (mbuf) operations
- GPUDev integration for GPU memory registration

Based on DPDK 23.11 LTS installed at /opt/dpdk/dpdk-23.11/install/
"""

from __future__ import annotations

import ctypes
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import IntEnum

logger = logging.getLogger(__name__)

# DPDK Constants
RTE_MAX_ETHPORTS = 32
RTE_ETHER_ADDR_LEN = 6
RTE_ETHER_TYPE_IPV4 = 0x0800
RTE_ETHER_TYPE_IPV6 = 0x86DD

class RteEthRxMqMode(IntEnum):
    """Ethernet device RX multi-queue mode"""
    ETH_MQ_RX_NONE = 0
    ETH_MQ_RX_RSS = 1
    ETH_MQ_RX_DCB = 2
    ETH_MQ_RX_DCB_RSS = 3
    ETH_MQ_RX_VMDQ_ONLY = 4
    ETH_MQ_RX_VMDQ_RSS = 5
    ETH_MQ_RX_VMDQ_DCB = 6
    ETH_MQ_RX_VMDQ_DCB_RSS = 7

class RteEthTxMqMode(IntEnum):
    """Ethernet device TX multi-queue mode"""
    ETH_MQ_TX_NONE = 0
    ETH_MQ_TX_DCB = 1
    ETH_MQ_TX_VMDQ_DCB = 2
    ETH_MQ_TX_VMDQ_ONLY = 3


def _load_first(names: List[str]) -> Optional[ctypes.CDLL]:
    """Load the first available library from a list of names with RTLD_GLOBAL."""
    for name in names:
        try:
            lib = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Successfully loaded DPDK library: {name}")
            return lib
        except OSError as e:
            logger.debug(f"Failed to load {name}: {e}")
            continue
    return None


@dataclass
class DpdkLibraries:
    """Container for all DPDK library handles."""
    eal: Optional[ctypes.CDLL] = None
    mempool: Optional[ctypes.CDLL] = None
    mbuf: Optional[ctypes.CDLL] = None
    ethdev: Optional[ctypes.CDLL] = None
    gpudev: Optional[ctypes.CDLL] = None

    @property
    def core_ok(self) -> bool:
        """Check if core libraries (EAL, mempool, mbuf) are available."""
        return all([self.eal, self.mempool, self.mbuf])

    @property
    def ethdev_ok(self) -> bool:
        """Check if ethernet device library is available."""
        return self.ethdev is not None

    @property
    def gpudev_ok(self) -> bool:
        """Check if GPU device library is available."""
        return self.gpudev is not None


def load_dpdk_libraries() -> DpdkLibraries:
    """Load all DPDK libraries with fallback paths."""
    dpdk_lib_path = "/opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu"
    
    # Try monolithic libdpdk first (best for symbol resolution)
    libdpdk = _load_first([
        f"{dpdk_lib_path}/libdpdk.so",
        "libdpdk.so",
        "/usr/lib/x86_64-linux-gnu/libdpdk.so",
    ])
    if libdpdk is not None:
        # Many symbols are available via the umbrella; reuse for all handles
        return DpdkLibraries(
            eal=libdpdk,
            mempool=libdpdk,
            mbuf=libdpdk,
            ethdev=libdpdk,
            gpudev=libdpdk,
        )
    
    # Core libraries
    eal = _load_first([
        f"{dpdk_lib_path}/librte_eal.so.24.0",
        f"{dpdk_lib_path}/librte_eal.so.24",
        f"{dpdk_lib_path}/librte_eal.so",
        "librte_eal.so", "librte_eal.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_eal.so",
    ])
    
    mempool = _load_first([
        f"{dpdk_lib_path}/librte_mempool.so.24.0",
        f"{dpdk_lib_path}/librte_mempool.so.24",
        f"{dpdk_lib_path}/librte_mempool.so",
        "librte_mempool.so", "librte_mempool.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_mempool.so",
    ])
    
    mbuf = _load_first([
        f"{dpdk_lib_path}/librte_mbuf.so.24.0",
        f"{dpdk_lib_path}/librte_mbuf.so.24",
        f"{dpdk_lib_path}/librte_mbuf.so",
        "librte_mbuf.so", "librte_mbuf.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_mbuf.so",
    ])
    
    # Ethernet device library
    ethdev = _load_first([
        f"{dpdk_lib_path}/librte_ethdev.so.24.0",
        f"{dpdk_lib_path}/librte_ethdev.so.24",
        f"{dpdk_lib_path}/librte_ethdev.so",
        "librte_ethdev.so", "librte_ethdev.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_ethdev.so",
    ])
    
    # GPU device library (optional)
    gpudev = _load_first([
        f"{dpdk_lib_path}/librte_gpudev.so.24.0",
        f"{dpdk_lib_path}/librte_gpudev.so.24",
        f"{dpdk_lib_path}/librte_gpudev.so",
        "librte_gpudev.so", "librte_gpudev.so.24",
        "/usr/lib/x86_64-linux-gnu/librte_gpudev.so",
    ])
    
    return DpdkLibraries(
        eal=eal,
        mempool=mempool,
        mbuf=mbuf,
        ethdev=ethdev,
        gpudev=gpudev
    )


class DpdkBindings:
    """Main DPDK bindings class with comprehensive functionality."""
    
    def __init__(self):
        self.libs = load_dpdk_libraries()
        self.eal_initialized = False
        self._default_pool = None
        self._port_info = {}
        
        if not self.libs.core_ok:
            logger.error("Failed to load core DPDK libraries")
            return
            
        self._setup_function_signatures()
        logger.info("DPDK bindings initialized successfully")

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for all DPDK functions."""
        try:
            self._setup_eal_functions()
            self._setup_mempool_functions()
            self._setup_mbuf_functions()
            if self.libs.ethdev_ok:
                self._setup_ethdev_functions()
            if self.libs.gpudev_ok:
                self._setup_gpudev_functions()
        except AttributeError as e:
            logger.error(f"Failed to setup function signatures: {e}")
            # Mark libraries as unavailable if signatures fail
            self.libs = DpdkLibraries()

    def _setup_eal_functions(self):
        """Setup EAL function signatures."""
        # int rte_eal_init(int argc, char **argv)
        self.libs.eal.rte_eal_init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.libs.eal.rte_eal_init.restype = ctypes.c_int
        
        # void rte_eal_cleanup(void)
        self.libs.eal.rte_eal_cleanup.argtypes = []
        self.libs.eal.rte_eal_cleanup.restype = None
        
        # unsigned rte_lcore_count(void)
        self.libs.eal.rte_lcore_count.argtypes = []
        self.libs.eal.rte_lcore_count.restype = ctypes.c_uint
        
        # unsigned rte_socket_count(void)
        self.libs.eal.rte_socket_count.argtypes = []
        self.libs.eal.rte_socket_count.restype = ctypes.c_uint

    def _setup_mempool_functions(self):
        """Setup mempool function signatures."""
        # struct rte_mempool* rte_pktmbuf_pool_create(...)
        self.libs.mbuf.rte_pktmbuf_pool_create.argtypes = [
            ctypes.c_char_p,    # name
            ctypes.c_uint,      # n (number of elements)
            ctypes.c_uint,      # cache_size
            ctypes.c_uint16,    # priv_size
            ctypes.c_uint16,    # data_room_size
            ctypes.c_int,       # socket_id
        ]
        self.libs.mbuf.rte_pktmbuf_pool_create.restype = ctypes.c_void_p

    def _setup_mbuf_functions(self):
        """Setup mbuf function signatures."""
        # struct rte_mbuf* rte_pktmbuf_alloc(struct rte_mempool *mp)
        self.libs.mbuf.rte_pktmbuf_alloc.argtypes = [ctypes.c_void_p]
        self.libs.mbuf.rte_pktmbuf_alloc.restype = ctypes.c_void_p
        
        # void rte_pktmbuf_free(struct rte_mbuf *m)
        self.libs.mbuf.rte_pktmbuf_free.argtypes = [ctypes.c_void_p]
        self.libs.mbuf.rte_pktmbuf_free.restype = None

    def _setup_ethdev_functions(self):
        """Setup ethernet device function signatures."""
        # uint16_t rte_eth_dev_count_avail(void)
        self.libs.ethdev.rte_eth_dev_count_avail.argtypes = []
        self.libs.ethdev.rte_eth_dev_count_avail.restype = ctypes.c_uint16
        
        # int rte_eth_dev_configure(uint16_t port_id, uint16_t nb_rx_q, uint16_t nb_tx_q, const struct rte_eth_conf *eth_conf)
        self.libs.ethdev.rte_eth_dev_configure.argtypes = [
            ctypes.c_uint16,    # port_id
            ctypes.c_uint16,    # nb_rx_q
            ctypes.c_uint16,    # nb_tx_q
            ctypes.c_void_p,    # eth_conf
        ]
        self.libs.ethdev.rte_eth_dev_configure.restype = ctypes.c_int
        
        # int rte_eth_dev_start(uint16_t port_id)
        self.libs.ethdev.rte_eth_dev_start.argtypes = [ctypes.c_uint16]
        self.libs.ethdev.rte_eth_dev_start.restype = ctypes.c_int
        
        # int rte_eth_dev_stop(uint16_t port_id)
        self.libs.ethdev.rte_eth_dev_stop.argtypes = [ctypes.c_uint16]
        self.libs.ethdev.rte_eth_dev_stop.restype = ctypes.c_int

    def _setup_gpudev_functions(self):
        """Setup GPU device function signatures."""
        # int rte_gpu_count_avail(void)
        self.libs.gpudev.rte_gpu_count_avail.argtypes = []
        self.libs.gpudev.rte_gpu_count_avail.restype = ctypes.c_int

    # Public API Methods
    
    def is_available(self) -> bool:
        """Check if DPDK bindings are available."""
        return self.libs.core_ok

    def init_eal(self, args: Optional[List[str]] = None) -> bool:
        """
        Initialize DPDK Environment Abstraction Layer.
        
        Args:
            args: Optional list of EAL arguments
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.libs.core_ok:
            logger.error("DPDK libraries not available")
            return False
            
        if self.eal_initialized:
            logger.info("EAL already initialized")
            return True
            
        # Default EAL arguments for our setup
        default_args = [
            "-c", "0xF",  # Use 4 cores
            "-n", "4",    # 4 memory channels
            "--huge-dir", "/mnt/huge",  # Hugepage directory
            "--proc-type", "primary",   # Primary process
        ]
        
        if args:
            eal_args = args
        else:
            eal_args = default_args
            
        # Prepare arguments
        argv = [b"genie-dpdk"] + [arg.encode("utf-8") for arg in eal_args]
        argc = len(argv)
        c_argv = (ctypes.c_char_p * argc)(*argv)
        
        logger.info(f"Initializing EAL with args: {[arg.decode() for arg in argv]}")
        ret = self.libs.eal.rte_eal_init(ctypes.c_int(argc), c_argv)
        
        if ret < 0:
            logger.error(f"EAL initialization failed with return code: {ret}")
            return False
            
        self.eal_initialized = True
        logger.info("EAL initialized successfully")
        
        # Log system information
        lcore_count = self.libs.eal.rte_lcore_count()
        socket_count = self.libs.eal.rte_socket_count()
        logger.info(f"DPDK initialized: {lcore_count} lcores, {socket_count} sockets")
        
        return True

    def cleanup_eal(self):
        """Cleanup EAL resources."""
        if self.eal_initialized and self.libs.eal:
            self.libs.eal.rte_eal_cleanup()
            self.eal_initialized = False
            logger.info("EAL cleanup completed")

    def create_mempool(self, name: str = "genie_pool", n_mbufs: int = 8192, 
                      cache_size: int = 256, data_room_size: int = 2048) -> Optional[ctypes.c_void_p]:
        """
        Create a packet buffer memory pool.
        
        Args:
            name: Pool name
            n_mbufs: Number of mbufs in the pool
            cache_size: Cache size per lcore
            data_room_size: Size of data room in each mbuf
            
        Returns:
            Memory pool pointer or None if failed
        """
        if not self.eal_initialized:
            logger.error("EAL not initialized")
            return None
            
        pool = self.libs.mbuf.rte_pktmbuf_pool_create(
            name.encode("utf-8"),
            ctypes.c_uint(n_mbufs),
            ctypes.c_uint(cache_size),
            ctypes.c_uint16(0),  # priv_size
            ctypes.c_uint16(data_room_size),
            ctypes.c_int(-1),  # socket_id (-1 = any socket)
        )
        
        if not pool:
            logger.error(f"Failed to create mempool '{name}'")
            return None
            
        if name == "genie_pool":
            self._default_pool = pool
            
        logger.info(f"Created mempool '{name}' with {n_mbufs} mbufs")
        return pool

    def get_default_pool(self) -> Optional[ctypes.c_void_p]:
        """Get the default memory pool."""
        return self._default_pool

    def alloc_mbuf(self, pool: Optional[ctypes.c_void_p] = None) -> Optional[ctypes.c_void_p]:
        """
        Allocate a packet buffer from memory pool.
        
        Args:
            pool: Memory pool to allocate from (uses default if None)
            
        Returns:
            Mbuf pointer or None if failed
        """
        if not self.eal_initialized:
            return None
            
        mp = pool or self._default_pool
        if not mp:
            logger.error("No memory pool available")
            return None
            
        mbuf = self.libs.mbuf.rte_pktmbuf_alloc(mp)
        if not mbuf:
            logger.warning("Failed to allocate mbuf")
            
        return mbuf

    def free_mbuf(self, mbuf: ctypes.c_void_p):
        """Free a packet buffer."""
        if mbuf:
            self.libs.mbuf.rte_pktmbuf_free(mbuf)

    def get_eth_dev_count(self) -> int:
        """Get number of available ethernet devices."""
        if not self.libs.ethdev_ok:
            return 0
        return self.libs.ethdev.rte_eth_dev_count_avail()

    def configure_eth_dev(self, port_id: int, nb_rx_queues: int = 1, nb_tx_queues: int = 1) -> bool:
        """
        Configure an ethernet device.
        
        Args:
            port_id: Port ID to configure
            nb_rx_queues: Number of RX queues
            nb_tx_queues: Number of TX queues
            
        Returns:
            True if successful, False otherwise
        """
        if not self.libs.ethdev_ok:
            logger.error("Ethdev library not available")
            return False
            
        # For now, pass NULL for eth_conf (use defaults)
        ret = self.libs.ethdev.rte_eth_dev_configure(
            ctypes.c_uint16(port_id),
            ctypes.c_uint16(nb_rx_queues),
            ctypes.c_uint16(nb_tx_queues),
            None  # eth_conf
        )
        
        if ret < 0:
            logger.error(f"Failed to configure port {port_id}: {ret}")
            return False
            
        logger.info(f"Configured port {port_id} with {nb_rx_queues} RX, {nb_tx_queues} TX queues")
        return True

    def start_eth_dev(self, port_id: int) -> bool:
        """Start an ethernet device."""
        if not self.libs.ethdev_ok:
            return False
            
        ret = self.libs.ethdev.rte_eth_dev_start(ctypes.c_uint16(port_id))
        if ret < 0:
            logger.error(f"Failed to start port {port_id}: {ret}")
            return False
            
        logger.info(f"Started ethernet port {port_id}")
        return True

    def stop_eth_dev(self, port_id: int) -> bool:
        """Stop an ethernet device."""
        if not self.libs.ethdev_ok:
            return False
            
        ret = self.libs.ethdev.rte_eth_dev_stop(ctypes.c_uint16(port_id))
        if ret < 0:
            logger.error(f"Failed to stop port {port_id}: {ret}")
            return False
            
        logger.info(f"Stopped ethernet port {port_id}")
        return True

    def get_gpu_count(self) -> int:
        """Get number of available GPU devices."""
        if not self.libs.gpudev_ok:
            return 0
        return self.libs.gpudev.rte_gpu_count_avail()

    def __del__(self):
        """Cleanup resources on destruction."""
        self.cleanup_eal()


# Global instance for easy access
_dpdk_instance: Optional[DpdkBindings] = None

def get_dpdk() -> DpdkBindings:
    """Get the global DPDK bindings instance."""
    global _dpdk_instance
    if _dpdk_instance is None:
        _dpdk_instance = DpdkBindings()
    return _dpdk_instance

def eal_init(args: Optional[List[str]] = None) -> bool:
    """Initialize DPDK EAL (convenience function)."""
    return get_dpdk().init_eal(args)

def is_dpdk_available() -> bool:
    """Check if DPDK is available (convenience function)."""
    return get_dpdk().is_available()
