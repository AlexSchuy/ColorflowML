import numpy as np
import os

import constants

def get_pixels(octet=False, recalculate=False):
  if octet:
    textfile = constants.octet_text
    npyfile = constants.octet_npy
    print("[data] Getting octet pixel data ...")
  else:
    textfile = constants.singlet_text
    npyfile = constants.singlet_npy
    print("[data] Getting singlet pixel data ...")
  # Get the appropriate numpy array, either from a saved .npy file or
  # from the original .txt file.
  if recalculate or not os.path.exists(npyfile):
    if not os.path.exists(npyfile):
      print("[data] {} not found, attempting to recreate from {} ...".format(npyfile, textfile))
    pixels = np.loadtxt(textfile, usecols=range(9, 634))
    # Remove any row with a nan value.
    if (np.isnan(pixels).any()):
      print("[data] Warning: non-numeric values encountered, removing affected samples.")
    pixels = pixels[~np.isnan(pixels).any(axis=1)]
    with open(npyfile, 'wb+') as npyfile_handle:
      np.save(npyfile_handle, pixels)
  else:
    print("[data] Loading from {} ...".format(npyfile))
    pixels = np.load(npyfile)
  print("[data] {} pixels shape: {}".format("octet" if octet else "singlet", pixels.shape))
  return pixels
