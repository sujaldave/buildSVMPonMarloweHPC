#!/bin/bash
#SBATCH --job-name=build-svmp
#SBATCH --partition=preempt
#SBATCH --account=marlowe-mXXXXXX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=14
#SBATCH --time=0-08:00:00
#SBATCH --output=build_svmp_%j.out
#SBATCH --error=build_svmp_%j.err
#SBATCH --no-requeue

# =============================================================================
# Build Boost, HDF5, LAPACK, Hypre, VTK, GPU-enabled Trilinos (Kokkos/CUDA,
# Hopper sm_90) and svMultiPhysics on Marlowe, as a single batch job.
#
# Usage (from a Marlowe login node):
#   1. Set --account (and --partition if needed) in the #SBATCH lines above.
#   2. Submit, choosing where everything goes:
#        sbatch --export=ALL,INSTALL_ROOT=/projects/<your-project>/<you>/svmp buildLibrariesAndSVMP.sh
#      If INSTALL_ROOT is not given, everything goes to
#        <directory you ran sbatch from>/svmp_marlowe
#
# Each stage writes a stamp file when it finishes. If the job is preempted
# or times out, submit it again with the same INSTALL_ROOT and it continues
# from the first unfinished stage. Delete $INSTALL_ROOT/stamps/<stage>.done
# to force a stage to be rebuilt.
# =============================================================================

set -eo pipefail

# ----------------------------- User settings ---------------------------------
INSTALL_ROOT="${INSTALL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/svmp_marlowe}"

# Parallel build jobs (defaults to the CPUs Slurm gave this job).
JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-24}}"

# svMultiPhysics source
SVMP_REPO="${SVMP_REPO:-https://github.com/sujaldave/svMultiPhysics.git}"
SVMP_BRANCH="${SVMP_BRANCH:-bipnWithTrilinos}"

# Library versions. An empty *_GIT_REF builds the repository's default branch
# (what the manual instructions do); set a tag/branch to pin a version, e.g.
#   sbatch --export=ALL,INSTALL_ROOT=...,TRILINOS_GIT_REF=trilinos-release-16-1-0 ...
BOOST_VERSION="${BOOST_VERSION:-1.89.0}"
VTK_VERSION="${VTK_VERSION:-9.3.0}"
HDF5_GIT_REF="${HDF5_GIT_REF:-}"
LAPACK_GIT_REF="${LAPACK_GIT_REF:-}"
HYPRE_GIT_REF="${HYPRE_GIT_REF:-}"
TRILINOS_GIT_REF="${TRILINOS_GIT_REF:-}"

# Marlowe modules and CUDA-aware Open MPI (NVIDIA HPC-X)
MODULES_BASE="slurm gcc/13.1.0 cmake/3.30.3"
MODULE_NVHPC="nvhpc/24.7"
HPCX_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19
MPI_HOME="${HPCX_HOME}/ompi"
# -----------------------------------------------------------------------------

SRC_DIR="${INSTALL_ROOT}/src"
BUILD_DIR="${INSTALL_ROOT}/build"
LIBS_DIR="${INSTALL_ROOT}/libs"
STAMP_DIR="${INSTALL_ROOT}/stamps"

BOOST_DIR="${LIBS_DIR}/boost"
HDF5_DIR="${LIBS_DIR}/hdf5"
LAPACK_DIR="${LIBS_DIR}/lapack"
HYPRE_DIR="${LIBS_DIR}/hypre"
VTK_DIR="${LIBS_DIR}/vtk"
TRILINOS_DIR="${LIBS_DIR}/trilinos_cuda_arch_hpr90"
SVMP_DIR="${INSTALL_ROOT}/svMultiPhysics"

MPICC="${MPI_HOME}/bin/mpicc"
MPICXX="${MPI_HOME}/bin/mpicxx"
MPIFORT="${MPI_HOME}/bin/mpifort"
MPIF90="${MPI_HOME}/bin/mpif90"
MPIRUN="${MPI_HOME}/bin/mpirun"

mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "${LIBS_DIR}" "${STAMP_DIR}"

# Make the 'module' command available in the batch shell if it is not already.
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        [[ -f "$f" ]] && source "$f" && break
    done
fi

module purge
# Clean PATH/LD_LIBRARY_PATH to return to before any MPI is added (used for VTK).
CLEAN_PATH="${PATH}"
CLEAN_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

log() { echo -e "\n[$(date '+%F %T')] ==== $* ====\n"; }

# Environment with only slurm + gcc + cmake (no CUDA-aware MPI). Used for VTK.
load_plain_env() {
    module purge
    module load ${MODULES_BASE}
    export PATH="${CLEAN_PATH}"
    export LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}"
}

# Environment with nvhpc and the CUDA-aware HPC-X Open MPI first in PATH.
load_cuda_mpi_env() {
    module purge
    module load ${MODULES_BASE} ${MODULE_NVHPC}
    export PATH="${MPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
}

# Clone a repo (optionally at a ref) into $SRC_DIR if it is not already there.
clone_repo() {
    local url="$1" dir="$2" ref="$3"
    if [[ ! -d "${SRC_DIR}/${dir}/.git" ]]; then
        rm -rf "${SRC_DIR:?}/${dir}"
        if [[ -n "${ref}" ]]; then
            git clone --branch "${ref}" "${url}" "${SRC_DIR}/${dir}"
        else
            git clone "${url}" "${SRC_DIR}/${dir}"
        fi
    fi
}

# run_stage <name> <function>: run the function in a subshell unless the stage is stamped.
run_stage() {
    local name="$1" fn="$2"
    if [[ -f "${STAMP_DIR}/${name}.done" ]]; then
        log "Skipping ${name} (already built; delete ${STAMP_DIR}/${name}.done to rebuild)"
        return
    fi
    log "Building ${name}"
    ( set -eo pipefail; "${fn}" )
    date > "${STAMP_DIR}/${name}.done"
    log "Finished ${name}"
}

# ------------------------------- Stages --------------------------------------

check_mpi() {
    load_cuda_mpi_env
    if ompi_info --parsable --all | grep -q "opal_built_with_cuda_support:value:true"; then
        echo "CUDA-aware Open MPI found: $(which mpicc)"
    else
        echo "ERROR: ${MPI_HOME} is not reporting CUDA support." >&2
        exit 1
    fi
}

build_boost() {
    load_cuda_mpi_env
    cd "${SRC_DIR}"
    local tag="boost_${BOOST_VERSION//./_}"
    if [[ ! -d "${tag}" ]]; then
        wget -q "https://archives.boost.io/release/${BOOST_VERSION}/source/${tag}.tar.gz"
        tar -xzf "${tag}.tar.gz" && rm -f "${tag}.tar.gz"
    fi
    cd "${tag}"
    mkdir -p "${BOOST_DIR}"
    ./bootstrap.sh --prefix="${BOOST_DIR}"
    ./b2 -j"${JOBS}" install
}

build_hdf5() {
    load_cuda_mpi_env
    clone_repo https://github.com/HDFGroup/hdf5.git hdf5 "${HDF5_GIT_REF}"
    rm -rf "${BUILD_DIR}/hdf5" && mkdir -p "${BUILD_DIR}/hdf5" "${HDF5_DIR}"
    cd "${BUILD_DIR}/hdf5"
    cmake -C "${SRC_DIR}/hdf5/config/cmake/cacheinit.cmake" -G "Unix Makefiles" \
        -DCMAKE_C_COMPILER="${MPICC}" \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_Fortran_COMPILER="${MPIFORT}" \
        -DMPIEXEC_EXECUTABLE="${MPIRUN}" \
        -DCMAKE_PREFIX_PATH="${MPI_HOME}" \
        -DCMAKE_INSTALL_RPATH="${MPI_HOME}/lib" \
        -DCMAKE_BUILD_RPATH="${MPI_HOME}/lib" \
        -DHDF5_ALLOW_UNSUPPORTED=ON \
        -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16:BOOL=OFF \
        -DHDF5_BUILD_JAVA:BOOL=OFF \
        -DHDF5_ENABLE_PARALLEL:BOOL=ON \
        -DALLOW_UNSUPPORTED:BOOL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${HDF5_DIR}" \
        "${SRC_DIR}/hdf5"
    cmake --build . --parallel "${JOBS}"
    make -j"${JOBS}" install
}

build_lapack() {
    load_cuda_mpi_env
    clone_repo https://github.com/Reference-LAPACK/lapack.git lapack "${LAPACK_GIT_REF}"
    rm -rf "${BUILD_DIR}/lapack" && mkdir -p "${BUILD_DIR}/lapack" "${LAPACK_DIR}"
    cd "${BUILD_DIR}/lapack"
    cmake \
        -DCMAKE_C_COMPILER="${MPICC}" \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_Fortran_COMPILER="${MPIFORT}" \
        -DMPIEXEC_EXECUTABLE="${MPIRUN}" \
        -DCMAKE_PREFIX_PATH="${MPI_HOME}" \
        -DCMAKE_INSTALL_RPATH="${MPI_HOME}/lib" \
        -DCMAKE_BUILD_RPATH="${MPI_HOME}/lib" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_LIBDIR="${LAPACK_DIR}" \
        "${SRC_DIR}/lapack"
    cmake --build . --parallel "${JOBS}"
    cmake --install . --config Release --prefix "${LAPACK_DIR}"
}

build_hypre() {
    load_cuda_mpi_env
    clone_repo https://github.com/hypre-space/hypre.git hypre "${HYPRE_GIT_REF}"
    cd "${SRC_DIR}/hypre/src"
    ./configure CC="${MPICC}" CXX="${MPICXX}" FC="${MPIFORT}" --prefix="${HYPRE_DIR}"
    make -j"${JOBS}" install
    # Trilinos looks for krylov.h, which newer hypre versions no longer install.
    printf '#include "HYPRE_krylov.h"\n' > "${HYPRE_DIR}/include/krylov.h"
}

build_vtk() {
    # VTK does not need the CUDA-aware MPI; build it with plain gcc.
    load_plain_env
    cd "${SRC_DIR}"
    if [[ ! -d "VTK-${VTK_VERSION}" ]]; then
        wget -q "https://www.vtk.org/files/release/${VTK_VERSION%.*}/VTK-${VTK_VERSION}.tar.gz"
        tar -xzf "VTK-${VTK_VERSION}.tar.gz" && rm -f "VTK-${VTK_VERSION}.tar.gz"
    fi
    rm -rf "${BUILD_DIR}/vtk" && mkdir -p "${BUILD_DIR}/vtk" "${VTK_DIR}"
    cd "${BUILD_DIR}/vtk"
    cmake \
        -DBUILD_SHARED_LIBS:BOOL=OFF \
        -DCMAKE_BUILD_TYPE:STRING=RELEASE \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF \
        -DVTK_USE_SYSTEM_EXPAT:BOOL=ON \
        -DVTK_USE_SYSTEM_ZLIB:BOOL=ON \
        -DVTK_LEGACY_REMOVE=ON \
        -DVTK_Group_Rendering=OFF \
        -DVTK_Group_StandAlone=OFF \
        -DVTK_RENDERING_BACKEND=None \
        -DVTK_WRAP_PYTHON=OFF \
        -DModule_vtkChartsCore=ON \
        -DModule_vtkCommonCore=ON \
        -DModule_vtkCommonDataModel=ON \
        -DModule_vtkCommonExecutionModel=ON \
        -DModule_vtkFiltersCore=ON \
        -DModule_vtkFiltersFlowPaths=ON \
        -DModule_vtkFiltersModeling=ON \
        -DModule_vtkIOLegacy=ON \
        -DModule_vtkIOXML=ON \
        -DVTK_GROUP_ENABLE_Views=NO \
        -DVTK_GROUP_ENABLE_Web=NO \
        -DVTK_GROUP_ENABLE_Imaging=NO \
        -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
        -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
        -DCMAKE_INSTALL_PREFIX="${VTK_DIR}" \
        "${SRC_DIR}/VTK-${VTK_VERSION}"
    cmake --build . --parallel "${JOBS}"
    make -j"${JOBS}" install
}

build_trilinos() {
    load_cuda_mpi_env
    clone_repo https://github.com/trilinos/Trilinos.git Trilinos "${TRILINOS_GIT_REF}"
    rm -rf "${BUILD_DIR}/trilinos" && mkdir -p "${BUILD_DIR}/trilinos" "${TRILINOS_DIR}"
    cd "${BUILD_DIR}/trilinos"
    cmake \
        -DCMAKE_INSTALL_PREFIX="${TRILINOS_DIR}" \
        -DTPL_ENABLE_MPI=ON \
        -DTPL_ENABLE_Boost=ON \
        -DBoost_LIBRARY_DIRS="${BOOST_DIR}/lib" \
        -DBoost_INCLUDE_DIRS="${BOOST_DIR}/include" \
        -DTPL_ENABLE_BLAS=ON \
        -DBLAS_LIBRARY_DIRS="${LAPACK_DIR}" \
        -DTPL_ENABLE_HDF5=ON \
        -DHDF5_LIBRARY_DIRS="${HDF5_DIR}/lib" \
        -DHDF5_INCLUDE_DIRS="${HDF5_DIR}/include" \
        -DTPL_ENABLE_HYPRE=ON \
        -DHYPRE_LIBRARY_DIRS="${HYPRE_DIR}/lib" \
        -DHYPRE_INCLUDE_DIRS="${HYPRE_DIR}/include" \
        -DTPL_ENABLE_LAPACK=ON \
        -DLAPACK_LIBRARY_DIRS="${LAPACK_DIR}" \
        -DCMAKE_C_COMPILER="${MPICC}" \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_Fortran_COMPILER="${MPIF90}" \
        -DTrilinos_ENABLE_MueLu=ON \
        -DTrilinos_ENABLE_ROL=ON \
        -DTrilinos_ENABLE_Sacado=ON \
        -DTrilinos_ENABLE_Teuchos=ON \
        -DTrilinos_ENABLE_Zoltan=ON \
        -DTrilinos_ENABLE_Tpetra=ON \
        -DTrilinos_ENABLE_Belos=ON \
        -DTrilinos_ENABLE_Ifpack2=ON \
        -DTrilinos_ENABLE_Amesos2=ON \
        -DTrilinos_ENABLE_Zoltan2=ON \
        -DTrilinos_ENABLE_Kokkos=ON \
        -DKokkos_ENABLE_SERIAL=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ARCH_HOPPER90=ON \
        -DTrilinos_ENABLE_KokkosKernels=ON \
        -DTrilinos_ENABLE_Xpetra=ON \
        -DXpetra_ENABLE_Kokkos_compat=ON \
        -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
        -DTpetra_INST_SERIAL=ON \
        -DTpetra_INST_DOUBLE=ON \
        -DTpetra_INST_INT_INT=ON \
        -DMueLu_ENABLE_EXPLICIT_INSTANTIATION=ON \
        -DTrilinos_ENABLE_Gtest=OFF \
        "${SRC_DIR}/Trilinos"
    make -j"${JOBS}" install
}

build_svmp() {
    load_cuda_mpi_env
    # svMultiPhysics finds Trilinos, LAPACK and VTK through these paths.
    export PATH="${TRILINOS_DIR}/lib/cmake/Trilinos:${LAPACK_DIR}:${VTK_DIR}:${PATH}"
    export CMAKE_PREFIX_PATH="${TRILINOS_DIR}:${VTK_DIR}:${HDF5_DIR}:${HYPRE_DIR}:${BOOST_DIR}:${CMAKE_PREFIX_PATH:-}"
    if [[ ! -d "${SVMP_DIR}/.git" ]]; then
        git clone -b "${SVMP_BRANCH}" "${SVMP_REPO}" "${SVMP_DIR}"
    fi
    rm -rf "${SVMP_DIR}/build" && mkdir -p "${SVMP_DIR}/build"
    cd "${SVMP_DIR}/build"
    cmake \
        -DSV_USE_TRILINOS:BOOL=ON \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_C_COMPILER="${MPICC}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -DNDEBUG2" \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -DNDEBUG2" \
        -DCMAKE_C_FLAGS="-DNDEBUG -DNDEBUG2" \
        ..
    make -j"${JOBS}"
    local exe="${SVMP_DIR}/build/svMultiPhysics-build/bin/svmultiphysics"
    [[ -x "${exe}" ]] || { echo "ERROR: ${exe} was not produced." >&2; exit 1; }
}

# Environment file that job scripts can source to run svMultiPhysics.
write_env_file() {
    cat > "${INSTALL_ROOT}/svmp_marlowe_env.sh" <<EOF
# Generated by buildLibrariesAndSVMP.sh on $(date '+%F %T').
# Usage in a job script:  source ${INSTALL_ROOT}/svmp_marlowe_env.sh
module purge
module load ${MODULES_BASE} ${MODULE_NVHPC}

export MPI_HOME=${MPI_HOME}
export PATH="\${MPI_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${MPI_HOME}/lib:\${LD_LIBRARY_PATH:-}"

export LIB_ROOT=${LIBS_DIR}
export PATH=\${LIB_ROOT}/trilinos_cuda_arch_hpr90/lib/cmake/Trilinos:\${PATH}
export PATH=\${LIB_ROOT}/lapack:\${PATH}
export PATH=\${LIB_ROOT}/vtk:\${PATH}
export LD_LIBRARY_PATH=\${LIB_ROOT}/lapack:\${LIB_ROOT}/hdf5/lib:\${LIB_ROOT}/hypre/lib:\${LIB_ROOT}/boost/lib:\${LD_LIBRARY_PATH}

export SVMP_EXE=${SVMP_DIR}/build/svMultiPhysics-build/bin/svmultiphysics
EOF
}

# --------------------------------- Run ---------------------------------------

log "Install root: ${INSTALL_ROOT} | parallel jobs: ${JOBS} | host: $(hostname)"

run_stage check_mpi check_mpi
run_stage boost     build_boost
run_stage hdf5      build_hdf5
run_stage lapack    build_lapack
run_stage hypre     build_hypre
run_stage vtk       build_vtk
run_stage trilinos  build_trilinos
run_stage svmp      build_svmp
write_env_file

# check_mpi is cheap; always re-run it on the next submission.
rm -f "${STAMP_DIR}/check_mpi.done"

log "All done"
echo "svMultiPhysics executable: ${SVMP_DIR}/build/svMultiPhysics-build/bin/svmultiphysics"
echo "Environment file for job scripts: ${INSTALL_ROOT}/svmp_marlowe_env.sh"
echo "Libraries installed under: ${LIBS_DIR}"
