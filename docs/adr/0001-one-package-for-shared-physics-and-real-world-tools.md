# One package for the shared physics and the real-world tools

slablet holds four layers: the plate-history contract, the response models that run on both synthetic worlds and
real reconstructions (mantle, asthenosphere, and intraplate volcanism moved in from IPV), the real-reconstruction
driver, and the tools that compare real runs with observations. We kept them in one private package, with import
tests enforcing the layers (the physics never imports pygplates, the driver or the observation tools), rather than
splitting them into separate packages, because the worst problem in the four repositories so far was a break at a
cross-repository seam (PlateDriver calling a `seed_slabs` signature that only slablet's uncommitted tree had), and
the layers change together almost daily. PlateDriver depends only on the contract and the physics; defamation, which
is real-only and mature, stays its own repository.

## Considered Options

- IPV as its own package importing the contract and driver from slablet: one more cross-repository seam.
- Separate packages for contract-plus-physics and driver-plus-observations: matches the two toolsets most literally,
  but splits code that changes together and adds a seam between them.
- Generic driver pieces in gprm: gprm is public and changes slowly; only a spherical rasterise function goes there.

## Consequences

The name slablet no longer describes the whole package; renaming is possible later. IPV's code has to reach slablet's
testing standard, and lose its shapely and geocube use, before it moves in. Details in
`docs/plans/2026-11-08-cross-repo-reconciliation.md`.
