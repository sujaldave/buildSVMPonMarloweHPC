#!/bin/bash
#SBATCH --job-name=svmp-test
#SBATCH --partition=preempt
#SBATCH --account=marlowe-mXXXXXX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=14
#SBATCH --time=0-00:30:00
#SBATCH --output=svmp_%j.out
#SBATCH --error=svmp_%j.err
#SBATCH --no-requeue

# Simple job script to test svMultiPhysics on Marlowe GPUs.
# One MPI rank per GPU. Change --ntasks-per-node and --gpus-per-node together.

# ---- Edit these ----
LIB_ROOT=/path/to/INSTALL_ROOT/libs
EXE=/path/to/INSTALL_ROOT/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics
XML=solver.xml
# --------------------

module purge
module load slurm gcc/13.1.0 cmake/3.30.3 nvhpc/24.7

# CUDA-aware Open MPI
export MPI_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"

# Libraries built by buildLibrariesAndSVMP.sh
export PATH=$LIB_ROOT/trilinos_cuda_arch_hpr90/lib/cmake/Trilinos:$PATH
export PATH=$LIB_ROOT/lapack:$PATH
export PATH=$LIB_ROOT/vtk:$PATH
export LD_LIBRARY_PATH=$LIB_ROOT/lapack:$LIB_ROOT/hdf5/lib:$LIB_ROOT/hypre/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1

mpirun -n "$SLURM_NTASKS" "$EXE" "$XML"
