import numpy
import os

import constants

def get_pixels(octet=False, recalculate=False):
  if octet:
    textfile = constants.octet_text
    npyfile = constants.octet_npy
  else:
    textfile = constants.singlet_text
    npyfile = constants.singlet_npy
  # Get the appropriate numpy array, either from a saved .npy file or
  # from the original .txt file.
  if recalculate or not os.path.exists(npyfile):
    with open(textfile) as textfile_handle:
      pixels = np.loadtxt(textfile_handle, usecols=range(9, 634))
    # Remove any row with a nan value.
    pixels = pixels[~np.isnan(pixels).any(axis=1)]
    with open(npyfile) as npyfile_handle:
      np.save(npyfile_handle, pixels)
  else:
    with open(npyfile) as npyfile_handle:
      pixels = np.load(npyfile_handle)
  return pixels
