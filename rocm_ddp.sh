# ROCm/HIP specific environment variables to fix IPC issues
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1  # Disable P2P communication (uses shared memory instead)
export HSA_FORCE_FINE_GRAIN_PCIE=1  # Force fine-grain memory for ROCm
# export NCCL_DEBUG=WARN  # Reduce debug output (change to INFO for debugging)
# export NCCL_SOCKET_IFNAME=eno1np0  # Use network interface from the logs

# Optional: Additional environment variables to optimize performance
# export NCCL_NET_GDR_LEVEL=0  # Disable GPU Direct RDMA
# export NCCL_SHM_DISABLE=1  # Disable shared memory (LAST RESORT ONLY)

NUM_GPUS=8

# Launch training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    cloud.py
