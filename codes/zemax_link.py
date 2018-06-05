import os
import numpy as np
import matplotlib.pyplot as plt
import pyzdde.zdde as pyz

# Create a DDE link object for communication
l1 = pyz.createLink()

# Load an existing file
zfile = os.path.join(l1.zGetPath()[1], 'Sequential',
                     'Objectives', 'Cooke 40 degree field.zmx')
l1.zLoadFile(zfile)

# Print out Surface Data Summary
l1.ipzGetLDE()

l1.ipzCaptureWindow('Lay')
