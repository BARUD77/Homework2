# CT artifact simulation: beam hardening + photon starvation with FBP
# This notebook generates a Shepp–Logan–like phantom with a metal insert,
# simulates projections under (1) monochromatic and (2) polychromatic spectra,
# adds Poisson noise (photon starvation), and reconstructs via FBP.
#
# Requirements: numpy, matplotlib, scikit-image.

import numpy as np
import matplotlib.pyplot as plt

# Try to import scikit-image; raise a helpful error if unavailable.
try:
    from skimage.data import shepp_logan_phantom
    from skimage.transform import radon, iradon, resize
except Exception as e:
    raise RuntimeError("This script requires scikit-image (skimage). Please install it to run the demo.") from e

# -----------------------------
# 1) Build phantom with metal
# -----------------------------
np.random.seed(0)
N = 256  # image size
phantom = shepp_logan_phantom()
phantom = resize(phantom, (N, N), mode="reflect", anti_aliasing=True)

# Normalize soft-tissue-ish background to ~0.02 mm^-1 (arbitrary units) at 70 keV
# and add a dense metal insert region (~5x attenuation of soft tissue).
mu_soft_70keV = 0.02
phantom = phantom * mu_soft_70keV

# Add a circular "metal" implant
yy, xx = np.mgrid[0:N, 0:N]
cx, cy, r = int(0.62*N), int(0.52*N), int(0.08*N)  # location & radius
metal_mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
mu_metal_70keV = 0.12  # much denser than soft tissue
phantom_with_metal = phantom.copy()
phantom_with_metal[metal_mask] = mu_metal_70keV

# -----------------------------
# 2) Define spectra and materials for forward model
# -----------------------------
# Discretize energy spectrum from 40 to 120 keV.
E = np.linspace(40, 120, 17)  # keV centers
# Simple tube spectrum-like weights (not physical, for illustration): triangular-ish
w = np.maximum(0, 1 - np.abs((E - 80) / 40))
w = w / w.sum()

# Energy-dependent attenuation models (toy but monotone-decreasing with E)
# Reference values anchored so that at 70 keV they match our chosen mu_*_70keV.
def mu_soft(Ekev):
    # ~ photoelectric + Compton-like trend; scale to hit 0.02 at 70 keV
    base = (70.0 / Ekev)**3 * 0.012 + 0.008
    # scale so mu_soft(70) == 0.02
    scale = mu_soft_70keV / ( (70.0 / 70.0)**3 * 0.012 + 0.008 )
    return scale * base

def mu_metal(Ekev):
    # steeper energy dependence; anchor to 0.12 at 70 keV
    base = (70.0 / Ekev)**3.5 * 0.35 + 0.02
    scale = mu_metal_70keV / ( (70.0 / 70.0)**3.5 * 0.35 + 0.02 )
    return scale * base

# -----------------------------
# 3) Generate sinograms
# -----------------------------
# Projection angles
angles = np.linspace(0., 180., 720, endpoint=False)

# Helper: compute line-length sinogram for soft tissue and metal separately
# We approximate material path lengths by summing pixel values of binary
