# Building svMultiPhysics with GPU-enabled Trilinos on Marlowe

This repository shows how to build **svMultiPhysics (svMP)** with a **GPU-enabled (Kokkos/CUDA) Trilinos** on the **Marlowe** cluster (Stanford, NVIDIA H100 GPUs), and how to run it.

Everything is built against the **CUDA-aware Open MPI** that ships with the NVIDIA HPC SDK (HPC-X) on Marlowe. The svMultiPhysics branch used is **`bipnWithTrilinos`**, which implements the **bi-partitioned (BIPN) linear solver with Tpetra operators**, so the solver can run on GPUs.

There are two ways to build:

1. **Automated (recommended):** submit one batch script, `buildLibrariesAndSVMP.sh`, with `sbatch`. It builds all the libraries and svMultiPhysics in a directory you choose. Marlowe does not offer interactive compute nodes, so the build runs as a job.
2. **Manual:** the same steps, with every command and flag, are listed in [Manual build](#manual-build-step-by-step) below. They are there for completeness and for anyone who wants to change a flag or debug a step.

> **Important**
> This workflow was tested on Marlowe with the modules listed below (`gcc/13.1.0`, `cmake/3.30.3`, `nvhpc/24.7`, HPC-X 2.19). Library repositories that are cloned from their default branch (HDF5, LAPACK, hypre, Trilinos) change over time. If a build breaks, pin a release (see [Pinning library versions](#pinning-library-versions)) and check that library's documentation.

---

## Repository contents

| File | Purpose |
|---|---|
| `buildLibrariesAndSVMP.sh` | Batch job that builds Boost, HDF5, LAPACK, hypre, VTK, Trilinos (CUDA, Hopper) and svMultiPhysics |
| `marloweSampleJobScript_simple.sh` | Minimal job script to test the solver; no communication tuning |
| `marloweSampleJobScript_singleNode.sh` | Tuned job script for one node (up to 8 GPUs) |
| `marloweSampleJobScript_multiNode.sh` | Tuned job script for more than one node |

---

## 1. Automated build (submit as a job)

### 1.1 Get the scripts on Marlowe

```bash
git clone https://github.com/sujaldave/buildSVMPonMarloweHPC.git
cd buildSVMPonMarloweHPC
```

### 1.2 Set your account

Open `buildLibrariesAndSVMP.sh` and change the `#SBATCH` lines at the top to your allocation:

```bash
#SBATCH --partition=preempt
#SBATCH --account=marlowe-mXXXXXX     # <-- your Marlowe account
```

The job asks for 1 node, 2 GPUs and 28 CPU cores for 8 hours. All 28 cores are used for compiling. Trilinos with CUDA is the slowest step.

### 1.3 Choose where to install and submit

The install location is set with `INSTALL_ROOT`. Pass it at submission time:

```bash
sbatch --export=ALL,INSTALL_ROOT=/projects/<your-project>/<your-name>/svmp_marlowe buildLibrariesAndSVMP.sh
```

If you do not pass `INSTALL_ROOT`, everything goes to `svmp_marlowe/` inside the directory you ran `sbatch` from.

Plan for several GB of disk space. A project directory (`/projects/...`) is usually a better choice than `$HOME`.

Follow progress with:

```bash
squeue -u $USER
tail -f build_svmp_<jobid>.out
```

### 1.4 If the job is preempted or runs out of time

Each stage (Boost, HDF5, LAPACK, hypre, VTK, Trilinos, svMP) writes a stamp file in `$INSTALL_ROOT/stamps/` when it finishes. **Submit the same command again** with the same `INSTALL_ROOT` and the job skips finished stages and restarts the unfinished one.

To force a stage to rebuild, delete its stamp, for example:

```bash
rm $INSTALL_ROOT/stamps/trilinos.done
```

### 1.5 Optional settings

All of these can be added to `--export=ALL,...`:

| Variable | Default | Meaning |
|---|---|---|
| `INSTALL_ROOT` | `$SLURM_SUBMIT_DIR/svmp_marlowe` | Where everything is installed |
| `JOBS` | CPUs given to the job | Parallel compile jobs |
| `SVMP_REPO` | `https://github.com/SimVascular/svMultiPhysics.git` | svMP repository (use your fork if needed) |
| `SVMP_BRANCH` | `bipnWithTrilinos` | svMP branch |
| `BOOST_VERSION` | `1.89.0` | Boost release |
| `VTK_VERSION` | `9.3.0` | VTK release |
| `HDF5_GIT_REF`, `LAPACK_GIT_REF`, `HYPRE_GIT_REF`, `TRILINOS_GIT_REF` | empty (default branch) | Tag or branch to check out |

#### Pinning library versions

For example, to build a specific Trilinos release:

```bash
sbatch --export=ALL,INSTALL_ROOT=/projects/<proj>/<you>/svmp_marlowe,TRILINOS_GIT_REF=<trilinos-release-tag> buildLibrariesAndSVMP.sh
```

### 1.6 What you get

```
$INSTALL_ROOT/
├── libs/                            # install prefixes
│   ├── boost/
│   ├── hdf5/
│   ├── lapack/
│   ├── hypre/
│   ├── vtk/
│   └── trilinos_cuda_arch_hpr90/
├── src/                             # downloaded sources
├── build/                           # out-of-source build directories
├── stamps/                          # one <stage>.done file per finished stage
├── svMultiPhysics/                  # svMP source, branch bipnWithTrilinos
│   └── build/svMultiPhysics-build/bin/svmultiphysics   # <-- executable
└── svmp_marlowe_env.sh              # environment to run svMP
```

`svmp_marlowe_env.sh` loads the modules and sets `PATH`/`LD_LIBRARY_PATH` for this install. It also sets `SVMP_EXE` to the path of the executable. You can `source` it in your own job scripts instead of copying the `export` lines.

---

## 2. Running svMultiPhysics

Three sample job scripts are provided. In each one, edit the block at the top:

```bash
LIB_ROOT=/path/to/INSTALL_ROOT/libs
EXE=/path/to/INSTALL_ROOT/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics
XML=solver.xml
```

and set `--account` in the `#SBATCH` lines. All three run **one MPI rank per GPU**, so keep `--ntasks-per-node` equal to `--gpus-per-node`.

| Script | Use it for | What it does |
|---|---|---|
| `marloweSampleJobScript_simple.sh` | A first test of the solver | Loads modules and runs `mpirun -n $SLURM_NTASKS`. No tuning. |
| `marloweSampleJobScript_singleNode.sh` | Production runs on **1 node (≤ 8 GPUs)** | UCX over shared memory and CUDA copies only (`UCX_TLS=self,sm,cuda_copy`). Pins one GPU and one core to each rank. |
| `marloweSampleJobScript_multiNode.sh` | Production runs on **2 or more nodes** | Adds the InfiniBand `rc` transport (`UCX_TLS=rc,sm,self,cuda_copy`) for traffic between nodes. Same GPU and core pinning. |

Submit with:

```bash
sbatch marloweSampleJobScript_singleNode.sh
```

For multi-node runs, set `--nodes`. The total number of ranks is `--nodes × --ntasks-per-node`, and the mesh must be partitioned for that many ranks.

---

## 3. Solver settings for the bi-partitioned solver with Trilinos

To use the bi-partitioned (BIPN) solver with Trilinos on the GPU, set the `<LS>` block in your `solver.xml` like this:

```xml
<LS type="NS" >
   <Linear_algebra type="trilinos" >
     <GMRES_Preconditioner> trilinos-resistance </GMRES_Preconditioner>
     <CG_Preconditioner> trilinos-diagonal </CG_Preconditioner>
   </Linear_algebra>
   <Max_iterations> 10 </Max_iterations>
   <NS_GM_max_iterations> 200 </NS_GM_max_iterations>
   <NS_CG_max_iterations> 500 </NS_CG_max_iterations>
   <Tolerance> 0.2 </Tolerance>
   <NS_GM_tolerance> 0.01 </NS_GM_tolerance>
   <NS_CG_tolerance> 0.02 </NS_CG_tolerance>
   <Krylov_space_dimension> 300 </Krylov_space_dimension>
</LS>
```

`<GMRES_Preconditioner>` and `<CG_Preconditioner>` set the preconditioners for the GMRES and CG solves **inside** the bi-partitioned solver. The recommended choices are:

| Problem | `<GMRES_Preconditioner>` | `<CG_Preconditioner>` |
|---|---|---|
| **CFD** | `trilinos-resistance` | `trilinos-diagonal` |
| **FSI** | `trilinos-ml` | `trilinos-diagonal` |

---

## Manual build (step by step)

These are the steps that `buildLibrariesAndSVMP.sh` runs, with every flag. Marlowe has no interactive compute nodes. Put these commands in a batch script, or run only light steps (downloads, `configure`) on a login node and follow the cluster's rules for compiling there.

In the commands below, set:

```bash
export INSTALL_ROOT=/path/of/your/choice          # e.g. /projects/<proj>/<you>/svmp_marlowe
export LIBS=$INSTALL_ROOT/libs
export MPI_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi
mkdir -p $INSTALL_ROOT/src $INSTALL_ROOT/build $LIBS
```

### M.1 Modules and CUDA-aware Open MPI

```bash
module purge
module load slurm
module load gcc/13.1.0 cmake/3.30.3 nvhpc/24.7

export PATH=$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
```

Check that this Open MPI was built with CUDA support:

```bash
ompi_info --parsable --all | grep "opal_built_with_cuda_support:value:true"
```

This must print a line. All libraries except VTK are compiled with this MPI's wrappers (`$MPI_HOME/bin/mpicc`, `mpicxx`, `mpifort`).

### M.2 Boost 1.89.0

```bash
cd $INSTALL_ROOT/src
wget https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
tar -xzvf boost_1_89_0.tar.gz && rm boost_1_89_0.tar.gz
cd boost_1_89_0
mkdir -p $LIBS/boost
./bootstrap.sh --prefix=$LIBS/boost && ./b2 install
```

### M.3 HDF5 (parallel)

```bash
cd $INSTALL_ROOT/src
git clone https://github.com/HDFGroup/hdf5.git
mkdir -p $INSTALL_ROOT/build/hdf5 $LIBS/hdf5 && cd $INSTALL_ROOT/build/hdf5

cmake -C $INSTALL_ROOT/src/hdf5/config/cmake/cacheinit.cmake -G "Unix Makefiles" \
  -DCMAKE_C_COMPILER=$MPI_HOME/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$MPI_HOME/bin/mpicxx \
  -DCMAKE_Fortran_COMPILER=$MPI_HOME/bin/mpifort \
  -DMPIEXEC_EXECUTABLE=$MPI_HOME/bin/mpirun \
  -DCMAKE_PREFIX_PATH=$MPI_HOME \
  -DCMAKE_INSTALL_RPATH=$MPI_HOME/lib \
  -DCMAKE_BUILD_RPATH=$MPI_HOME/lib \
  -DHDF5_ALLOW_UNSUPPORTED=ON \
  -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16:BOOL=OFF \
  -DHDF5_BUILD_JAVA:BOOL=OFF \
  -DHDF5_ENABLE_PARALLEL:BOOL=ON \
  -DALLOW_UNSUPPORTED:BOOL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$LIBS/hdf5 \
  $INSTALL_ROOT/src/hdf5

cmake --build . && make -j24 install
```

### M.4 LAPACK / BLAS (shared)

```bash
cd $INSTALL_ROOT/src
git clone https://github.com/Reference-LAPACK/lapack.git
mkdir -p $INSTALL_ROOT/build/lapack $LIBS/lapack && cd $INSTALL_ROOT/build/lapack

cmake \
  -DCMAKE_C_COMPILER=$MPI_HOME/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$MPI_HOME/bin/mpicxx \
  -DCMAKE_Fortran_COMPILER=$MPI_HOME/bin/mpifort \
  -DMPIEXEC_EXECUTABLE=$MPI_HOME/bin/mpirun \
  -DCMAKE_PREFIX_PATH=$MPI_HOME \
  -DCMAKE_INSTALL_RPATH=$MPI_HOME/lib \
  -DCMAKE_BUILD_RPATH=$MPI_HOME/lib \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_LIBDIR=$LIBS/lapack \
  $INSTALL_ROOT/src/lapack

cmake --build . && cmake --install . --config Release --prefix $LIBS/lapack
```

`CMAKE_INSTALL_LIBDIR` is an absolute path, so `libblas.so` and `liblapack.so` end up directly in `$LIBS/lapack`. Trilinos expects them there.

Add to `~/.bashrc`:

```bash
export PATH=$LIBS/lapack:$PATH
```

### M.5 hypre

```bash
cd $INSTALL_ROOT/src
git clone https://github.com/hypre-space/hypre.git && cd hypre/src
./configure CC=$MPI_HOME/bin/mpicc CXX=$MPI_HOME/bin/mpicxx FC=$MPI_HOME/bin/mpifort \
  --prefix=$LIBS/hypre
make -j24 install
```

### M.6 VTK 9.3.0 (no CUDA-aware MPI needed)

VTK is built with plain `gcc`. Start from a **fresh login** (log out and in again) so nothing from the HPC-X Open MPI is left in `PATH`. Then load only:

```bash
module purge
module load slurm gcc/13.1.0 cmake/3.30.3
```

```bash
cd $INSTALL_ROOT/src
wget https://www.vtk.org/files/release/9.3/VTK-9.3.0.tar.gz
tar -xzvf VTK-9.3.0.tar.gz && rm VTK-9.3.0.tar.gz
mkdir -p $INSTALL_ROOT/build/vtk $LIBS/vtk && cd $INSTALL_ROOT/build/vtk

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
  -DCMAKE_INSTALL_PREFIX=$LIBS/vtk \
  $INSTALL_ROOT/src/VTK-9.3.0

cmake --build . --parallel 24 && make -j24 install
```

Add to `~/.bashrc`:

```bash
export PATH=$LIBS/vtk:$PATH
```

### M.7 Trilinos with Kokkos/CUDA for H100 (Hopper, sm_90)

Reload the CUDA-aware environment from [M.1](#m1-modules-and-cuda-aware-open-mpi) first.

Newer hypre versions no longer install the `krylov.h` header that Trilinos includes. Create it:

```bash
printf '#include "HYPRE_krylov.h"\n' > $LIBS/hypre/include/krylov.h
```

Then:

```bash
cd $INSTALL_ROOT/src
git clone https://github.com/trilinos/Trilinos.git
mkdir -p $INSTALL_ROOT/build/trilinos $LIBS/trilinos_cuda_arch_hpr90 && cd $INSTALL_ROOT/build/trilinos

cmake \
  -DCMAKE_INSTALL_PREFIX=$LIBS/trilinos_cuda_arch_hpr90 \
  -DTPL_ENABLE_MPI=ON \
  -DTPL_ENABLE_Boost=ON \
  -DBoost_LIBRARY_DIRS=$LIBS/boost/lib \
  -DBoost_INCLUDE_DIRS=$LIBS/boost/include \
  -DTPL_ENABLE_BLAS=ON \
  -DBLAS_LIBRARY_DIRS=$LIBS/lapack \
  -DTPL_ENABLE_HDF5=ON \
  -DHDF5_LIBRARY_DIRS=$LIBS/hdf5/lib \
  -DHDF5_INCLUDE_DIRS=$LIBS/hdf5/include \
  -DTPL_ENABLE_HYPRE=ON \
  -DHYPRE_LIBRARY_DIRS=$LIBS/hypre/lib \
  -DHYPRE_INCLUDE_DIRS=$LIBS/hypre/include \
  -DTPL_ENABLE_LAPACK=ON \
  -DLAPACK_LIBRARY_DIRS=$LIBS/lapack \
  -DCMAKE_C_COMPILER=$MPI_HOME/bin/mpicc \
  -DCMAKE_CXX_COMPILER=$MPI_HOME/bin/mpicxx \
  -DCMAKE_Fortran_COMPILER=$MPI_HOME/bin/mpif90 \
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
  $INSTALL_ROOT/src/Trilinos

make -j24 install
```

Add to `~/.bashrc`:

```bash
export PATH=$LIBS/trilinos_cuda_arch_hpr90/lib/cmake/Trilinos:$PATH
```

svMultiPhysics' CMake finds Trilinos, LAPACK and VTK through these `PATH` entries.

### M.8 svMultiPhysics (branch `bipnWithTrilinos`)

Use the CUDA-aware environment from M.1 and the `PATH` entries from M.4, M.6 and M.7:

```bash
cd $INSTALL_ROOT
git clone -b bipnWithTrilinos https://github.com/SimVascular/svMultiPhysics.git
cd svMultiPhysics && mkdir build && cd build

cmake \
  -DSV_USE_TRILINOS:BOOL=ON \
  -DCMAKE_CXX_COMPILER=$MPI_HOME/bin/mpicxx \
  -DCMAKE_C_COMPILER=$MPI_HOME/bin/mpicc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -DNDEBUG2" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -DNDEBUG2" \
  -DCMAKE_C_FLAGS="-DNDEBUG -DNDEBUG2" \
  ..

make -j24
```

The executable is:

```
$INSTALL_ROOT/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics
```

---

## Troubleshooting

- **`opal_built_with_cuda_support` is not `true`:** the wrong `mpicc` is first in `PATH`. Check with `which mpicc`. It should be under `$MPI_HOME/bin`.
- **VTK fails to configure or link against MPI:** HPC-X is still in your environment. Start a fresh shell and load only `slurm gcc cmake`.
- **Trilinos cannot find `krylov.h`:** create it as shown in [M.7](#m7-trilinos-with-kokkoscuda-for-h100-hopper-sm_90).
- **A default-branch library suddenly fails to build:** pin it to a release tag (`*_GIT_REF` for the automated build, `git checkout <tag>` for the manual one).
- **Multi-node run hangs or reports no transport between nodes:** make sure `UCX_TLS` includes `rc`, as in `marloweSampleJobScript_multiNode.sh`.
