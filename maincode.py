import pandas as pd
import numpy as np

# 1. LOAD DATA
df = pd.read_csv('radial_z01_data.csv')  #reading the csv data file 
df = df[['radius','temp','h2','h2o','o2']] #selecting specific columns
df = df.sort_values('radius').reset_index(drop=True) #sorting the data based on radius

# Calcuting the local equivalence ratio 

MW_H2 = 2.016
MW_O2 = 32.0
stoich_mass_ratio = 7.936  # kg O2 per kg H2 for H2–air

# avoid divide-by-zero
df['o2_safe'] = df['o2'].replace(0, 1e-12) #replace zero O2 values with a small number to avoid division by zero in a fuel rich area

# φ = (Y_H2/Y_O2) / (1/stoich_mass_ratio)
df['phi'] = (df['h2'] / df['o2_safe']) / (1.0 / stoich_mass_ratio) #calculating equivalence ratio

# 3. STRATIFICATION METRICS

# gradient dφ/dr
df['dphi_dr'] = np.gradient(df['phi'], df['radius'])

SI = df['phi'].std() / df['phi'].mean()          # stratification index
max_grad = abs(df['dphi_dr']).max()              # max |dφ/dr|
r_max_grad = df.loc[abs(df['dphi_dr']).idxmax(),'radius']

print(f"Mean φ      = {df['phi'].mean():.3f}")
print(f"SI          = {SI:.3f}")
print(f"Max |dφ/dr| = {max_grad:.1f} 1/m at r = {r_max_grad*1000:.1f} mm")

# 4. SIMPLE LOCAL EFFICIENCY (using H2 depletion)

Y_H2_inlet = df['h2'].max()          # approximate inlet H2 mass fraction
df['eta_local'] = (Y_H2_inlet - df['h2']) / Y_H2_inlet * 100.0
df['eta_local'] = df['eta_local'].clip(0, 100)

print(f"Mean η_local = {df['eta_local'].mean():.1f} %")

# 5. SAVE PROCESSED DATA FOR PLOTS
df.to_csv('radial_case1_processed.csv', index=False)
