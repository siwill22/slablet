#!/bin/bash
# Build the local Geode Archive for comparing PlateDriver mantle runs with Müller 2022 OPT1 and present-day
# tomography (REVEAL, SEMUCB-WM1), 0-1000 Ma, in the Müller 2022 optimised mantle frame (anchor 0).
#
#   bash reconstructions/build_geode_archive.sh            # writes runs_real/geode_archive (gitignored)
#
# Needs a Geode checkout at $GEODE (default ~/GIT/Geode) whose prep/prep_convection.py has --volume-var, and
# Geode's own archive for REVEAL, SEMUCB-WM1, colormaps and the surface image.  View it with:
#   ln -sfn $PWD/runs_real/geode_archive $GEODE/viewer/public/archive-platedriver
#   cd $GEODE/viewer && VITE_ARCHIVE_BASE=/archive-platedriver npx vite --port 5180
# then open http://localhost:5180/ and pick the preset "Kinematic model vs OPT1 vs tomography".
set -e
GEODE=${GEODE:-$HOME/GIT/Geode}
R=$(cd "$(dirname "$0")/.." && pwd)/runs_real
A=$R/geode_archive
M22=$HOME/Library/Caches/gprm/Muller2022
PY="conda run -n pygmt17 python"
mkdir -p $A/models
cp $GEODE/archive/colormaps.json $A/
cp -R $GEODE/archive/surface $A/
cp -R $GEODE/archive/models/reveal $GEODE/archive/models/semucb $A/models/
cd $GEODE

# our runs: one 3-D NetCDF per age, kept at their native 5 deg x 29 depths; dVs, high = fast, shown at +-3 %
run () {
  $PY prep/prep_convection.py --input "$2" --volume-var dVs --age-regex '_(\d+)Ma' --age-max 1000 \
    --id "$1" --name "$3" --source "$4" --var-id vs --var-name "dVs (kinematic model)" --units % --high-means fast \
    --colormap RdBu --nlon 72 --nlat 37 --ndepth 29 --default-clip -3 3 --out $A --validate
}
run pd-dip-pile $R/muller2022_dip_pile_1000 "PlateDriver mantle: dip + pile" \
  "PlateDriver kinematic mantle (configs/real/muller2022_dip_pile_1000.yaml), Muller 2022 optimised frame"
run pd-dip-nopile $R/muller2022_dip_slabs_1000 "PlateDriver mantle: dip, no pile" \
  "PlateDriver kinematic mantle (configs/real/muller2022_dip_slabs_1000.yaml), Muller 2022 optimised frame"
run pd-opt1-best $R/sweeps_opt1/o1_halo_profile/s2_tau100 "PlateDriver mantle: OPT1-tuned (slowing 2, CMB tau 100)" \
  "PlateDriver kinematic mantle, sweep o1_halo_profile variant s2_tau100, Muller 2022 optimised frame"
run pd-opt1-sharp $R/sweeps_opt1/o1_halo_profile/s2_tau100_halo300_sharp "PlateDriver mantle: OPT1-tuned, halo 300 km, no blur" \
  "PlateDriver kinematic mantle, sweep o1_halo_profile variant s2_tau100_halo300_sharp, Muller 2022 optimised frame"

# the slab-width-profile run, before the tomography blur (dVs_native), at its native 1 deg (as nodes)
$PY prep/prep_convection.py --input $R/muller2022_dip_pile_sharp --volume-var dVs_native --age-regex '_(\d+)Ma' \
  --age-max 1000 --id pd-dip-pile-sharp --name "PlateDriver mantle: dip + pile, sharp slabs (1 deg)" \
  --source "PlateDriver kinematic mantle (configs/real/muller2022_dip_pile_sharp.yaml, dVs_native), Muller 2022 optimised frame" \
  --var-id vs --var-name "dVs (kinematic model)" --units % --high-means fast --colormap RdBu \
  --nlon 360 --nlat 181 --ndepth 29 --default-clip -3 3 --out $A --validate

# OPT1, 0-1000 Ma every 20 Myr, at Geode's standard 1 deg x 192 depths
$PY prep/prep_convection.py --input $HOME/Data/zenodo/OPT1_temperature_anomaly_grids_dimensional --id opt1 \
  --name "Muller 2022 OPT1" --age-max 1000 --out $A --validate \
  --source "Muller et al. 2022, optimised mantle convection model (OPT1); temperature anomaly, horizontal mean removed"

# Müller 2022 coastlines in the optimised mantle frame, 0-1000 Ma at 1 Myr
$PY prep/prep_coastlines.py --coastlines $M22/Coastlines/shapes_coastlines_Merdith_etal.gpmlz \
  --rotations $M22/optimisation/1000_0_rotfile_MantleOpt.rot --age-min 0 --age-max 1000 --age-step 1 --anchor 0 \
  --out $A/coastlines \
  --source "Muller et al. (2022) coastlines (Merdith et al.) and optimised mantle-frame rotations (1000_0_rotfile_MantleOpt.rot), anchor 0"

$PY prep/build_archive_index.py --archive $A --age-min 0 --age-max 1000
