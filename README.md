# Hotspots

The "hotspots" repository is currently under development and aims to provide the functionality to detect 
potential thunderstorm updrafts based on polarimetric radar measurements. Details about the algorithm can be found 
in Krause and Klaus (2024): https://doi.org/10.1175/WAF-D-23-0146.1 and in Klaus and Krause (2024): https://doi.org/10.1175/WAF-D-23-0227.1

Recent development was done by John Krause (john.krause@noaa.gov) and Vinzent Klaus (vinzent.klaus@boku.ac.at). We 
would be more than happy to offer additional help. If you're interested in contributing to the development of an 
open-source updraft detection package for Python, please reach out to us.

## Quick start

We recommend installing a conda environment based on the provided environment.yml. As we do not provide a proper automatic 
setup yet, please add the "hotspots" directory to your conda search path, for example, using the "conda develop" command.

### Environment setup
The conda environment used in the hotspots repository can be created by the command:

`$conda env create -f environment.yml`

This command takes some time to build the hotspots environment, but it only has to be run once.

Then use:

`$conda activate hotspots`

when finished running the hotspot scripts:

`$conda deactivate` 

### Quick overview
The hotspot detection is implemented in the "hotspots" submodule. The standard version in "hotspots_xy.py" takes a 
CAPPI on a Cartesian grid as input. The CAPPI must be available as xarray.Dataset and contain (1) ZH, (2) ZDR and 
(3) DR (circular depolarization ratio; proxy variable based on ZDR and RHOHV). The hotspot detection is performed on 
the ZDR field, which requires preprocessing to cut it down to the region of interest (i.e., ZDR close to areas of 
ZH > 25 dBZ). We recommend using a CAPPI level corresponding to the environmental -10°C isothermal height, derived from radiosonde or NWP. This ensures that we are above the melting level in an environment where positive ZDR anomalies caused by ZDR 
columns are really prominent. 

The workflow from CAPPIs of quality-controlled ZH, ZDR and derived DR to the final hotspot field consists 
of the following steps:

    zdrcut = hotspots.detection.get_filtered_zdr(
        xycappi, refl_fieldname='prepro_zh', zdr_fieldname='prepro_zdr',
        cdr_fieldname='prepro_dr')

    xycappi["zdr_cut"] = zdrcut

    hotspot_field_proc, hotspot_features_proc = (
        hotspots.detection.apply_hotspot_method(
        xycappi, x_dim="x", y_dim="y", refl_fieldname='prepro_zh',
        zdr_fieldname='zdr_cut'))

The code above provides the hotspot field and the detected hotspot features. Note that the hotspot features are 
indicative of thunderstorm updrafts, but there are still many artifacts left in the data. The results from the hotspot 
detection must be coupled to a cell tracking system to filter out hotspots that are not linked to thunderstorm cells.

We have included python code and scripts to compute cells based on the MCIT method proposed by: Rosenfeld (1987) and Hu, J., and Coauthors, (2019): https://doi.org/10.1016/j.atmosres.2019.04.015 . This output is coupled to a simple overlap based tracking system of our own design. The cell identification and tracking system can be used to assign hotspots to storm cells and to remove those hotspots that remain unassigned as proposed in Klaus and Krause, (2024): https://journals.ametsoc.org/view/journals/wefo/39/12/WAF-D-23-0227.1.xml . The code to accomplish these tasks can be found in the hotspots/scripts directory. 

### Full overview

Hotspot detection relies on CAPPI's in ZH, ZDR and DR. When developing the hotspot algorithm we used code in C++ to create and display the output. However, we note that many in the community work in python and we have endevored to convert our C++ code into python. As such the python code base is a close approximation of our work in C++, but not exact. We've encountered and overcome a few signigicant obsticles in creating the repository and hotspots outputs. 

The first obsticle is out ability to make the CAPPI data itself. While the pyart repository is used to read the Level 2 format used in USA NWS formated data and to compute the DR variable, we did not find it useful in creating the CAPPI outputs we needed. To do so we first had to convert the data from Az/Ran format to XY format. This is handled in our implementation by the use of the xarray.Dataset class. We found the xarray module which contains the Dataarray and Dataset classes so useful that we used it in the entire repository. You can find our methods to convert Az/Ran/Elev data to XY format in the cappi submodule, which uses the excellent work in wradlib.

Another obsticle we encountered was the lack of preprocessing of the raw radar data. You will note in the quick overview that we have "prepro" as part of the variable names, as this data went through our preprocessing system. A set of preprocessing tasks was created because there are several procedures that need to be applied to the raw radar data to prepare that data for general use. There are many scientific thoughts on how to properly prepare the data, but we generally followed the method used on the USA NWS radar network. This includes tasks like applying a running average to reduce the variablity of the data, unfolding phidp, and correcting for horizontal attenuation. Unfortunately data comparisons between our python based approach and the operational outputs suggest that we have more work to do. However, you are free to skip the prepro methods or substitue your own. Radar data from the non-NWS networks may have their own procedures or data from a multi-radar network might already be in CAPPI format. We (John.Krause@noaa.gov) are happy to explore other preprocessing tasks or the improvement of the ones supplied.

Finally we note that the examples use data from the USA NWS radar network NEXRAD which is an S-band based collection of radars. The main reason for this is that that data from this nework are freely available in multiple locations (e.g., https://s3.amazonaws.com/noaa-nexrad-level2/index.html). Our original work used data from this network exclusively, but we have always wanted to make an algorithm that was useful on any radar system. We have designed the code to be modular in that you can start the processing at any point where you feel your data meets the requirements, either after preprocessing or if you already have created CAPPI's. We hope that it can even be used with model generated data. We are actively solicitating data from other radars to demonstrate the hotspot technique.  

## Future plans

We wish to improve the gridding/interpolation/quality control routines that are also part of the package 
(submodules "cappi" and "prepro"). These should highlight a potential way to prepare the data for the hotspot detection.

Improving the cell tracking algorithm or identifying a way to tie into a more accurate algorithm is on our todo list.






```python

```


