# global_mesopelagic_respiration
Code to reproduce the results from Gómez-Letona &amp; Álvarez-Salgado (2026). Steps:

1. Install the conda environment specified in the `.yml` file (`oceanicu.yml`). Set the project folder as the current directory (e.g., `cd MY/PATH/TO/FOLDER/global_mesopelagic_respiration-main`) and type the following command in the terminal:
   
   ```
   conda env create -f oceanicu.yml
   ```

   If you need help, check [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

   Activate the environment:
   ```
   conda activate oceanicu
   ```
2. Before running the project code a very minor edit is required to download POC data, see details in `scripts/download_data/04_download_poc3d.py`.
3. Run:
   ```
   python scripts/master.py > logs.txt
   ```

> [!NOTE]
> (3) Alternatively, for an interactive execution launch the Spyder IDE (type `spyder` in the terminal) and then open the [project](https://docs.spyder-ide.org/current/panes/projects.html) from the the parent directory. Make sure that in `Tools > Preferences > Working directory` the selected option is `The project ...`, and in `Tools > Preferences > Run` within the configuration for the `.py` extension the `Working directory settings` has `The current working directory` selected. Otherwise relative paths won't work. Note: this works for Spyder 6.1, might be different for previous versions. Execute the `master.py` script.


### Data availability
The source data for this work are publicly available:
-  [**GLODAPv2 (main dataset)**](https://glodap.info/index.php/merged-and-adjusted-data-product-v2-2023/).
-  [Hansell et al. (2021) (DOM)](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0227166).
-  [Atmospheric histories of CFC-11, CFC-12, SF6](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0164584).
-  [NPP](http://orca.science.oregonstate.edu/npp_products.php).
-  [WOA](https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23.pl).
-  [POC](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_BIO_BGC_3D_REP_015_010/description). (Complementary data for potential future work, not used in the article; only covers the upper 1000 m)

*Note: scripts to automatically download the source data are part of the code provided here.*

The results produced in this work are available at:
-  [OUR estimates for the global mesopelagic ocean](https://doi.org/10.5281/zenodo.15801466).
-  [Compilation of OURs from previous works](https://doi.org/10.5281/zenodo.15801341).

### Citation
Gómez‐Letona, M. (2026). Code for: Respiration rates in the global mesopelagic ocean (Version 1.0.0). Zenodo. [https://doi.org/10.5281/zenodo.19222382](https://doi.org/10.5281/zenodo.19222382)

### Acknowledgments
This work was funded by the European Union under grant agreement no. 101083922 (OceanICU) and UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10054454, 10063673, 10064020, 10059241, 10079684, 10059012, 10048179]. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
<br /><img src="https://github.com/markeletona/global_mesopelagic_respiration/blob/main/OceanICU_logo_1.4.png" width="300" height="109" />

