import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# configuring the xcel file path and output folder 
excel_file = 'case3_data.xlsx'
output_folder = 'Case3_Results'
os.makedirs(output_folder, exist_ok=True)

# axial positions, with the corresponding excel sheets (standardised)
axial_positions = {
    '1': 0.025,   # 25mm - Near field
    '2': 0.05,    # 50mm - Near field  
    '3': 0.15,    # 150mm - Far field
    '4': 0.25,    # 250mm - Far field
    '5': 0.35     # 350mm - Far field
}

NEAR_FIELD = [0.025, 0.05]
FAR_FIELD = [0.15, 0.25, 0.35]

# extracting the data and putting it into a dataframe 
xls = pd.ExcelFile(excel_file)
results = []

for sheet_name, z_m in axial_positions.items():
    z_mm = z_m * 1000
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['radius'] < 0.1].copy()
    df = df[df['h2o'] > 0.001].copy()
    
    mean_h2o = df['h2o'].mean()
    std_h2o = df['h2o'].std()
    SI_h2o = std_h2o / mean_h2o
    
    TKE = df['tke'].mean()
    TI = df['ti'].mean()
    
    regime = 'Near' if z_m in NEAR_FIELD else 'Far'
    
    results.append({
        'z_mm': z_mm,
        'z_m': z_m,
        'regime': regime,
        'SI_h2o': SI_h2o,
        'mean_h2o': mean_h2o,
        'std_h2o': std_h2o,
        'TKE': TKE,
        'TI': TI
    })

df_results = pd.DataFrame(results)

# calculating stratification gradients for near and far fields using H2O data 
near_field = df_results[df_results['regime'] == 'Near'].copy()
far_field = df_results[df_results['regime'] == 'Far'].copy()

# mean strafitication gradients 
gradient_near = (near_field['SI_h2o'].iloc[-1] - near_field['SI_h2o'].iloc[0]) / (near_field['z_mm'].iloc[-1] - near_field['z_mm'].iloc[0])
gradient_far = (far_field['SI_h2o'].iloc[-1] - far_field['SI_h2o'].iloc[0]) / (far_field['z_mm'].iloc[-1] - far_field['z_mm'].iloc[0])

print("="*80)
print("STRATIFICATION GRADIENT ANALYSIS")
print("="*80)
print(f"Near-field dSI/dz: {gradient_near:.6f} mm⁻¹")
print(f"Far-field dSI/dz:  {gradient_far:.6f} mm⁻¹")
print("="*80)

# plotting the results 
# plot 1 : SI vs Distance 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

ax1.plot(df_results['z_mm'], df_results['SI_h2o'], 'k-', linewidth=2)
ax1.scatter(near_field['z_mm'], near_field['SI_h2o'], s=200, c='red', marker='o', edgecolors='black', linewidths=2, label='Near-field', zorder=3)
ax1.scatter(far_field['z_mm'], far_field['SI_h2o'], s=200, c='blue', marker='s', edgecolors='black', linewidths=2, label='Far-field', zorder=3)
ax1.set_xlabel('Axial Distance [mm]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Stratification Index (SI)', fontsize=12, fontweight='bold')
ax1.set_title('SI vs Distance', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# plot 2 : TKE vs Distance
ax2.plot(df_results['z_mm'], df_results['TKE'], 'g-', linewidth=2)
ax2.scatter(near_field['z_mm'], near_field['TKE'], s=200, c='red', marker='o', edgecolors='black', linewidths=2, label='Near-field', zorder=3)
ax2.scatter(far_field['z_mm'], far_field['TKE'], s=200, c='blue', marker='s', edgecolors='black', linewidths=2, label='Far-field', zorder=3)
ax2.set_xlabel('Axial Distance [mm]', fontsize=12, fontweight='bold')
ax2.set_ylabel('TKE [m²/s²]', fontsize=12, fontweight='bold')
ax2.set_title('TKE vs Distance', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# plot 3: SI vs TKE - Near Field
ax3.plot(near_field['TKE'], near_field['SI_h2o'], 'r-', linewidth=2)
ax3.scatter(near_field['TKE'], near_field['SI_h2o'], s=200, c='red', marker='o', edgecolors='black', linewidths=2, zorder=3)
ax3.set_xlabel('TKE [m²/s²]', fontsize=12, fontweight='bold')
ax3.set_ylabel('Stratification Index (SI)', fontsize=12, fontweight='bold')
ax3.set_title('SI vs TKE - Near Field', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# plot 4: SI vs TKE - Far Field
ax4.plot(far_field['TKE'], far_field['SI_h2o'], 'b-', linewidth=2)
ax4.scatter(far_field['TKE'], far_field['SI_h2o'], s=200, c='blue', marker='s', edgecolors='black', linewidths=2, zorder=3)
ax4.set_xlabel('TKE [m²/s²]', fontsize=12, fontweight='bold')
ax4.set_ylabel('Stratification Index (SI)', fontsize=12, fontweight='bold')
ax4.set_title('SI vs TKE - Far Field', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/Analysis_Results.png', dpi=300)
plt.show()

# saving the results to excel files
summary = {
    'Near_field_dSI_dz': gradient_near,
    'Far_field_dSI_dz': gradient_far,
    'Initial_SI': df_results.iloc[0]['SI_h2o'],
    'Final_SI': df_results.iloc[-1]['SI_h2o']
}

summary_df = pd.DataFrame([summary])
summary_df.to_excel(f'{output_folder}/Summary.xlsx', index=False)
df_results.to_excel(f'{output_folder}/All_Data.xlsx', index=False)

print(f"\n✓ Plots saved to: {output_folder}/Analysis_Results.png")
print(f"✓ Data saved to: {output_folder}/All_Data.xlsx")

output_excel = f'{output_folder}/Complete_Analysis_Data.xlsx'

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: Summary
    summary = {
        'Near_field_dSI_dz': [gradient_near],
        'Far_field_dSI_dz': [gradient_far],
        'Initial_SI': [df_results.iloc[0]['SI_h2o']],
        'Final_SI': [df_results.iloc[-1]['SI_h2o']]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Sheet 2: All positions data
    formatted_results = pd.DataFrame({
        'Axial Position [mm]': df_results['z_mm'],
        'Regime': df_results['regime'],
        'Stratification Index (SI)': df_results['SI_h2o'],
        'Mean H2O': df_results['mean_h2o'],
        'Std H2O': df_results['std_h2o'],
        'TKE [m²/s²]': df_results['TKE'],
        'Turbulence Intensity [%]': df_results['TI']
    })
    formatted_results.to_excel(writer, sheet_name='All_Positions', index=False)
    
    # Individual sheets for each position
    for idx, row in df_results.iterrows():
        position_data = pd.DataFrame({
            'Parameter': ['Axial Position [mm]', 'Regime', 'Stratification Index (SI)', 
                         'Mean H2O', 'Std H2O', 'TKE [m²/s²]', 'Turbulence Intensity [%]'],
            'Value': [row['z_mm'], row['regime'], row['SI_h2o'], 
                     row['mean_h2o'], row['std_h2o'], row['TKE'], row['TI']]
        })
        sheet_name = f'z={int(row["z_mm"])}mm'
        position_data.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n✓ Plots saved to: {output_folder}/Analysis_Results.png")
print(f"✓ All data saved to: {output_excel}")
