import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv('radial_z01_data.csv')  #reading the csv data file 
df = df[['radius','temp','h2','h2o','o2']] #selecting specific columns
# Calcuting the local equivalence ratio 

MW_H2 = 2.016
MW_O2 = 32.0
stoich_mass_ratio = 7.936  # kg O2 per kg H2 for H2–air

# avoid divide-by-zero
df['o2_safe'] = df['o2'].replace(0, 1e-12) #replace zero O2 values with a small number to avoid division by zero in a fuel rich area

# φ = (Y_H2/Y_O2) / (1/stoich_mass_ratio)
df['phi'] = (df['h2'] / df['o2_safe']) / (1.0 / stoich_mass_ratio) #calculating equivalence ratio

df['dphi_dr'] = np.gradient(df['phi'], df['radius']) #numerical gradient of phi with respect to radius 

SI = df['phi'].std() / df['phi'].mean()          # stratification index which is basically just the coefficient of variation
max_grad = abs(df['dphi_dr']).max()              # max |dφ/dr| helps clearly identify steepness of gradients
r_max_grad = df.loc[abs(df['dphi_dr']).idxmax(),'radius'] # radius at which max |dφ/dr| occurs 

print(f"Mean φ      = {df['phi'].mean():.3f}")
print(f"SI          = {SI:.3f}")
print(f"Max |dφ/dr| = {max_grad:.1f} 1/m at r = {r_max_grad*1000:.1f} mm")

#calculating local effieciency

Y_H2_inlet = df['h2'].max()          #because we use dry H2 at the inlet, the max H2 mass fraction is the inlet value 
df['eta_local'] = (Y_H2_inlet - df['h2']) / Y_H2_inlet * 100.0 #local efficiency in percentage
df['eta_local'] = df['eta_local'].clip(0, 100) #ensures that efficiency values are between 0 and 100%

print(f"Mean η_local = {df['eta_local'].mean():.1f} %")

#saving processed data to a new csv file 
df.to_csv('radial_case1_processed.csv', index=False)


#---------------------------------------------------------------------

# Setup
r_mm = df['radius'] * 1000  # Convert m to mm for prettier graphs

# Create the 2x2 Grid
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3) # Spacing

# ---------------------------------------------------------
# PLOT 1: Temperature & Fuel (Top Left)
# ---------------------------------------------------------
ax[0,0].plot(r_mm, df['temp'], 'r-', linewidth=2, label='Temp')
ax[0,0].set_xlabel('Radius [mm]')
ax[0,0].set_ylabel('Temperature [K]', color='r')
ax[0,0].tick_params(axis='y', labelcolor='r')
ax[0,0].grid(True, alpha=0.3)
ax[0,0].set_title('(a) Flame Structure')

# Dual Axis for Fuel
ax2 = ax[0,0].twinx()
ax2.plot(r_mm, df['h2'], 'b--', linewidth=1.5, label='H2 Mass Frac')
ax2.set_ylabel('H2 Mass Fraction', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# ---------------------------------------------------------
# PLOT 2: Equivalence Ratio (Top Right)
# ---------------------------------------------------------
ax[0,1].plot(r_mm, df['phi'], 'k-', linewidth=2)
ax[0,1].axhline(y=1.0, color='g', linestyle='--', label='Stoichiometric')
ax[0,1].set_xlabel('Radius [mm]')
ax[0,1].set_ylabel('Equivalence Ratio ($\phi$)')
ax[0,1].set_title('(b) Mixing Quality')
ax[0,1].grid(True, alpha=0.3)
ax[0,1].legend()

# Optional: Limit Y-axis if phi is huge at inlet
# ax[0,1].set_ylim(0, 5) 

# ---------------------------------------------------------
# PLOT 3: Stratification Gradient (Bottom Left)
# ---------------------------------------------------------
ax[1,0].plot(r_mm, abs(df['dphi_dr']), 'purple', linewidth=2)
ax[1,0].set_xlabel('Radius [mm]')
ax[1,0].set_ylabel('Stratification Gradient $|d\phi/dr|$ [m$^{-1}$]')
ax[1,0].set_title('(c) Shear Layer Intensity')
ax[1,0].grid(True, alpha=0.3)

# ---------------------------------------------------------
# PLOT 4: Efficiency (Bottom Right)
# ---------------------------------------------------------
ax[1,1].plot(r_mm, df['eta_local'], 'g-', linewidth=2)
ax[1,1].fill_between(r_mm, df['eta_local'], color='g', alpha=0.1) # Shading looks nice
ax[1,1].set_xlabel('Radius [mm]')
ax[1,1].set_ylabel('Local Efficiency $\eta$ [%]')
ax[1,1].set_title('(d) Combustion Performance')
ax[1,1].set_ylim(0, 105)
ax[1,1].grid(True, alpha=0.3)

# Save
plt.suptitle(f"Case Analysis: S={0.0} $\phi$={1.0}", fontsize=16) # Change title per case
plt.savefig('QuadChart_Case1.png', dpi=300)
plt.show()
