#!/bin/bash
#SBATCH --job-name=svmp-multinode
#SBATCH --partition=preempt
#SBATCH --account=marlowe-mXXXXXX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=14
#SBATCH --time=0-02:00:00
#SBATCH --output=svmp_multinode_%j.out
#SBATCH --error=svmp_multinode_%j.err
#SBATCH --no-requeue
###SBATCH --mail-user=<you>@stanford.edu
###SBATCH --mail-type=ALL

# Optimized for more than one Marlowe node (8 H100 GPUs per node).
# One MPI rank per GPU: total ranks = --nodes x --ntasks-per-node.
# Ranks on different nodes talk over InfiniBand, so UCX needs the 'rc'
# transport in addition to shared memory + CUDA copies.

set -euo pipefail

# ---- Edit these ----
LIB_ROOT=/path/to/INSTALL_ROOT/libs
EXE=/path/to/INSTALL_ROOT/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics
XML=solver.xml
# --------------------

module purge
module load slurm gcc/13.1.0 cmake/3.30.3 nvhpc/24.7

# CUDA-aware Open MPI (NVIDIA HPC-X)
export MPI_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

# Libraries built by buildLibrariesAndSVMP.sh
export PATH=$LIB_ROOT/trilinos_cuda_arch_hpr90/lib/cmake/Trilinos:$PATH
export PATH=$LIB_ROOT/lapack:$PATH
export PATH=$LIB_ROOT/vtk:$PATH
export LD_LIBRARY_PATH=$LIB_ROOT/lapack:$LIB_ROOT/hdf5/lib:$LIB_ROOT/hypre/lib:$LD_LIBRARY_PATH

MPI_RANKS="${SLURM_NTASKS}"
RANKS_PER_NODE="${SLURM_NTASKS_PER_NODE}"

export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Multi-node communication configuration.
export UCX_TLS=rc,sm,self,cuda_copy
export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=n
export UCX_WARN_UNUSED_ENV_VARS=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^openib

[[ -x "${EXE}" ]] || { echo "ERROR: executable not found or not executable: ${EXE}"; exit 1; }
[[ -f "${XML}" ]] || { echo "ERROR: XML file not found: ${XML}"; exit 1; }

echo "Nodes:      ${SLURM_JOB_NODELIST}"
echo "MPI ranks:  ${MPI_RANKS} (${RANKS_PER_NODE} per node)"
echo "Executable: ${EXE}"
echo "Input:      ${XML}"
echo "UCX_TLS:    ${UCX_TLS}"

# Each rank sees only the GPU matching its node-local rank.
mpirun -n "${MPI_RANKS}" \
    --map-by ppr:${RANKS_PER_NODE}:node:PE=1 \
    --bind-to core \
    --report-bindings \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x OMP_NUM_THREADS \
    -x CUDA_DEVICE_ORDER \
    -x UCX_TLS \
    -x UCX_MEMTYPE_CACHE \
    -x UCX_IB_GPU_DIRECT_RDMA \
    -x UCX_WARN_UNUSED_ENV_VARS \
    bash -c '
        export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK}"
        echo "rank=${OMPI_COMM_WORLD_RANK} local_rank=${OMPI_COMM_WORLD_LOCAL_RANK} gpu=${CUDA_VISIBLE_DEVICES}"
        exec "$@"
    ' bash "${EXE}" "${XML}"
