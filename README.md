# Vaccination and transportation intervention strategies for effective pandemic control
- This repository contains the raw data and programming file for the preprint manuscript submitted to Transport Policy.
  - Liu Yang, Kashin Sugishita, Shinya Hanaoka, Vaccination and transportation interventionstrategies for effective pandemic control, 2024.
- Initial version May 3, 2024; last updated May 3, 2024. 
- Due to certain confidential issues, some raw data in the folder are preprocessed.  

## Programs files
- Data read & preprocess files reads csv data from V_data floder and process to pkl files for main program. It also plots the accutal infection data and vaccination data.
- VR_SDP file is the calculating file for the optimization problem. The results are stored in Results folder in .pkl format. 
- Figures file plots the figures using results and raw data.
- Two .py file are moudules built for calculating state variables, adjoint variables and total costs.


## V_data Folder
- Contains raw data collected from official websites.
- .pkl files are preprocessed files for better reading.

## Results Folder
- Storing the results data from main programing file VR_SDP.ipynb.
- Please see the naming rules from VR_SDP.ipynb file.

## gadm36_JPN_shp Folder
- This file is for the map ploting in Figures.ipynb.
