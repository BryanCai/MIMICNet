import numpy as np
import pandas as p
import os
import pdb

codes = p.read_csv('../Data/codes.csv')
diastolic = p.read_csv('../Data/diastolic.csv')
FiO2 = p.read_csv('../Data/FiO2.csv')
GCS = p.read_csv('../Data/GCS.csv')
glucose = p.read_csv('../Data/glucose.csv')
hr = p.read_csv('../Data/hr.csv')
O2Saturation = p.read_csv('../Data/O2Saturation.csv')
pH = p.read_csv('../Data/pH.csv')
systolic = p.read_csv('../Data/systolic.csv')
urine = p.read_csv('../Data/urine.csv')

data = [diastolic, FiO2, GCS, glucose, hr, O2Saturation, pH, systolic, urine]

for d in data:
	d.rename(columns={'?column?': 'time'}, inplace=True)
