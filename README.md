# multitaskrepresentations
#### Demo code repository for Ito &amp; Murray (2023)
#### Contacts: taku.ito1@gmail.com; john.murray@yale.edu
#### Last updated: 1/2/2023

Citation: Ito T, Murray JD (2023). Multitask representations in the human cortex transform along a sensory-to-motor hierarchy. Nature Neuroscience. https://www.nature.com/articles/s41593-022-01224-0

## Overview

This code repository provides demo code and derivative (processed) data to generate all main-text figures. It also provides some processed data for select Supplementary Figures (for smaller files). This repository includes all raw code that was used for analyses and modeling on the publicly available Multi-Domain Task Battery dataset. Some of the code is written for the Yale cluster, but in principle, the python and shell scripts can be adapted for servers/clusters.

Link to the Multi-Domain Task Battery dataset: [https://openneuro.org/datasets/ds002105](https://openneuro.org/datasets/ds002105)
Citation for the Multi-Domain Task Battery dataset:
King, M., Hernandez-Castillo, C.R., Poldrack, R.A., Ivry, R.B., Diedrichsen, J., 2019. Functional boundaries in the human cerebellum revealed by a multi-domain task battery. Nature Neuroscience 22, 1371–1378. [https://doi.org/10.1038/s41593-019-0436-x](https://doi.org/10.1038/s41593-019-0436-x)


## Description/Organization of code repository

`code/`: contains both Jupyter Notebooks that generate Figure panels from derivative processed data. Note that while all processed data for main text figures are included, not all derivative data for supplementary figure panels is included (due to file size constraints). However, all the code required to generate the derivative data is included (in `*.py` files). Note that `*.py` files are included, and generate derivative data from preprocessed fMRI data.

`figures/`: contains all figure panels.

`processed_data/`: contains processed (or derivative) data required to generate figure panels in the Jupyter Notebooks. Data can be accessed [here](https://drive.google.com/drive/folders/1ooZgxGzVtkgHmeZz7BZI79qcYML2XFwX?usp=sharing).

<!---
`code/preprocessing/`: contains all preprocessing scripts that processed the raw MDTB dataset from OpenNeuro. Preprocessing was performed using QuNex (version 0.61.17; [https://qunex.yale.edu/](https://qunex.yale.edu/)). Postprocessing (i.e., task activation estimation) can be found in `code/preprocessing/glm_scripts/`. Note that all preprocessing scripts are provided as-is, as they were all performed on Yale's compute cluster. Preprocessing scripts are provided as a reference for those who wish to adapt them to their needs. Any questions can be directed to taku.ito1@gmail.com.
-->
