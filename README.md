# POP_slicer

Physical Optics Propagation (POP) analysis of Image Slicers for ELT-HARMONI and ELT-PCS.

## Methodology

POP analysis is done using Zemax. The output is a set of Zemax Beam Files (ZBF), one for each of the Slices within the Image Slicer. Those files are loaded with Python and post-processed to generate a single PSF.
![Alt text](sample.png?raw=true "Title")

Current analyses include: the effect of oversizing the pupil mirrors to minimize light loss due to diffraction at the image slicer, the influence of that oversize on the RMS wavefront error.

## Machine Learning capabilities
Using Zemax POP we can generate thousands of example PSFs with varying NCPA phase maps. These PSFs include the effect of the image slicer on the light propagation so they are fairly realistic representations of what happens at the slicer.

We can use those PSFs to train a Neural Network so that it can recognize the Zernike coefficients of the underlying NCPA phase maps just by looking at the image PSFs. 

In order to avoid any degeneracy in the transformation from phase map to PSF intensity, we repeat the POP analysis for each phase including an additional defocus term. Thus, the images we pass to the NN containg pairs of _nominal_ and _defocused_ slicer PSFs, just like the one shown below.

![Alt text](examplePSF.png?raw=true "Title")
