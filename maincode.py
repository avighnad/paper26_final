import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS - Change these for your analysis
# =============================================================================
data_file = 'case1_data.xlsx'
output_folder = 'Case1'  # Folder to store all output files

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# axial sheets from excel 
axial_positions = {
    '1': 0.025,  # 25mm - near injection
    '2': 0.050,  # 50mm - early flame
    '3': 0.100,  # 100mm - mid-flame
    '4': 0.200,  # 200mm - late combustion
    '5': 0.300   # 300mm - post-flame
}

# local stoichiometric ratio for H2-O2 combustion 
STOICH_RATIO = 7.936  # kg O2 per kg H2

# Global H2 inlet mass fraction (boundary condition in CFD)
GLOBAL_H2_INLET = 1.0  # Pure H2 at fuel injectors

# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================
print("="*70)
print("PROCESSING ALL AXIAL POSITIONS")
print("="*70)

# Load the Excel file
xls = pd.ExcelFile(data_file)
all_results = []

# Process each sheet
for sheet_name, z_position in axial_positions.items():
    
    if sheet_name not in xls.sheet_names:
        continue
    
    print(f"\n📍 Processing z = {z_position*1000:.0f} mm (Sheet '{sheet_name}')")
    
    # loading and preparing data 
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    df = df[['radius', 'temp', 'h2o', 'h2', 'o2']].copy()
    
    # calculating the equivalence ratio and its gradient 
    # Avoid division by zero in fuel-rich regions
    df['o2_safe'] = df['o2'].replace(0, 1e-12)
    
    # φ = (H2/O2)_actual / (H2/O2)_stoich
    df['phi'] = (df['h2'] / df['o2_safe']) / (1.0 / STOICH_RATIO)
    
    # calculate how phi changes with radius
    df['dphi_dr'] = np.gradient(df['phi'], df['radius'])
    
    # stratification metric
    mean_phi = df['phi'].mean()
    std_phi = df['phi'].std()
    
    # stratification Index = std/mean (coefficient of variation)
    SI = std_phi / mean_phi
    
    # rms for gradient (emphasizes large gradients)
    rms_gradient = np.sqrt(np.mean(df['dphi_dr']**2))
    
    # mean absolute gradient (average magnitude)
    mean_gradient = np.mean(np.abs(df['dphi_dr']))
    
    # Find the steepest gradient location
    max_gradient = abs(df['dphi_dr']).max()
    max_gradient_location = df.loc[abs(df['dphi_dr']).idxmax(), 'radius'] * 1000  # mm
    
    # local combustion efficiency based on H2 consumption 
    # Using global inlet H2 mass fraction (1.0) as reference for all axial positions
    
    # Efficiency = how much H2 has been consumed relative to inlet
    df['eta_local'] = ((GLOBAL_H2_INLET - df['h2']) / GLOBAL_H2_INLET * 100).clip(0, 100)
    
    mean_efficiency = df['eta_local'].mean()
    
    # flame location
    # Temperature gradient shows where the flame front is
    df['dT_dr'] = np.gradient(df['temp'], df['radius'])
    
    # calculating where the temp is the highest, and also where the temperature is changed the most rapidly
    max_temp = df['temp'].max()
    r_at_max_temp = df.loc[df['temp'].idxmax(), 'radius'] * 1000  # mm 
    r_at_max_temp_grad = df.loc[abs(df['dT_dr']).idxmax(), 'radius'] * 1000  # mm
    
    # saving the processed data to csv
    case_name = f'case1_z{int(z_position*1000):03d}mm'
    df.to_csv(os.path.join(output_folder, f'{case_name}_processed.csv'), index=False)
    
    # writing a text file with summary of findings for ease 
    with open(os.path.join(output_folder, f'{case_name}_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"ANALYSIS SUMMARY: {case_name.upper()}\n")
        f.write(f"Axial Position: z = {z_position*1000:.0f} mm\n")
        f.write("="*60 + "\n\n")
        
        f.write("EQUIVALENCE RATIO\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean φ               : {mean_phi:.3f}\n")
        f.write(f"Std Dev φ            : {std_phi:.3f}\n")
        f.write(f"Min φ                : {df['phi'].min():.3f}\n")
        f.write(f"Max φ                : {df['phi'].max():.3f}\n\n")
        
        f.write("STRATIFICATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Stratification Index : {SI:.3f}\n")
        f.write(f"Mean |dφ/dr|         : {mean_gradient:.1f} m⁻¹\n")
        f.write(f"RMS gradient         : {rms_gradient:.1f} m⁻¹\n")
        f.write(f"Max gradient         : {max_gradient:.1f} m⁻¹\n")
        f.write(f"  at radius          : {max_gradient_location:.2f} mm\n\n")
        
        f.write("EFFICIENCY\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean η_local         : {mean_efficiency:.1f} %\n")
        f.write(f"Max η_local          : {df['eta_local'].max():.1f} %\n")
        f.write(f"Min η_local          : {df['eta_local'].min():.1f} %\n\n")
        
        f.write("TEMPERATURE\n")
        f.write("-"*40 + "\n")
        f.write(f"Max Temperature      : {max_temp:.1f} K\n")
        f.write(f"Min Temperature      : {df['temp'].min():.1f} K\n")
        f.write(f"Mean Temperature     : {df['temp'].mean():.1f} K\n\n")
        
        f.write("FLAME CHARACTERISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Flame location (T_max)       : {r_at_max_temp:.2f} mm\n")
        f.write(f"Flame location (max dT/dr)   : {r_at_max_temp_grad:.2f} mm\n\n")
        
        f.write("="*60 + "\n")
    
    # Store results for comparison
    all_results.append({
        'z_mm': z_position * 1000,
        'mean_phi': mean_phi,
        'SI': SI,
        'mean_grad': mean_gradient,
        'rms_grad': rms_gradient,
        'max_grad': max_gradient,
        'mean_eta': mean_efficiency,
        'max_temp': max_temp,
        'r_flame_temp': r_at_max_temp,
        'r_flame_grad': r_at_max_temp_grad
    })
    
    print(f"   ✓ Mean φ = {mean_phi:.3f}, SI = {SI:.3f}, η = {mean_efficiency:.1f}%")

# =============================================================================
# CREATE MASTER SUMMARY FILE
# =============================================================================
print("\n" + "="*70)
print("Creating master summary file...")

with open(os.path.join(output_folder, 'case1_master_summary.txt'), 'w') as f:
    f.write("="*90 + "\n")
    f.write("MASTER SUMMARY - ALL AXIAL POSITIONS\n")
    f.write("="*90 + "\n\n")
    
    # Table 1: Main metrics at each location
    f.write("MAIN METRICS\n")
    f.write("-"*100 + "\n")
    f.write(f"{'z [mm]':<10} {'Mean φ':<10} {'SI':<10} {'Mean ∇φ':<12} {'RMS ∇φ':<12} "
            f"{'Max ∇φ':<12} {'η [%]':<10} {'T_max [K]':<12}\n")
    f.write("-"*100 + "\n")
    
    for r in all_results:
        f.write(f"{r['z_mm']:<10.0f} {r['mean_phi']:<10.3f} {r['SI']:<10.3f} "
                f"{r['mean_grad']:<12.1f} {r['rms_grad']:<12.1f} {r['max_grad']:<12.1f} "
                f"{r['mean_eta']:<10.1f} {r['max_temp']:<12.1f}\n")
    
    # Table 2: Evolution rates (how fast things change downstream)
    f.write("\n\nEVOLUTION RATES (per 100mm downstream)\n")
    f.write("-"*100 + "\n")
    f.write(f"{'z [mm]':<10} {'ΔSI/Δz':<12} {'Δ(Mean∇φ)/Δz':<15} {'Δ(RMS∇φ)/Δz':<15} "
            f"{'Δη/Δz [%]':<15} {'ΔT_max/Δz [K]':<15}\n")
    f.write("-"*100 + "\n")
    
    for i in range(1, len(all_results)):
        # Calculate change per 100mm
        dz = (all_results[i]['z_mm'] - all_results[i-1]['z_mm']) / 100
        
        dSI = (all_results[i]['SI'] - all_results[i-1]['SI']) / dz
        dMean = (all_results[i]['mean_grad'] - all_results[i-1]['mean_grad']) / dz
        dRMS = (all_results[i]['rms_grad'] - all_results[i-1]['rms_grad']) / dz
        dEta = (all_results[i]['mean_eta'] - all_results[i-1]['mean_eta']) / dz
        dTemp = (all_results[i]['max_temp'] - all_results[i-1]['max_temp']) / dz
        
        f.write(f"{all_results[i]['z_mm']:<10.0f} {dSI:<12.4f} {dMean:<15.2f} {dRMS:<15.2f} "
                f"{dEta:<15.2f} {dTemp:<15.1f}\n")
    
    # Table 3: Flame location evolution
    f.write("\n\nFLAME LOCATION\n")
    f.write("-"*90 + "\n")
    f.write(f"{'z [mm]':<10} {'r_flame (T_max) [mm]':<25} {'r_flame (∇T) [mm]':<25}\n")
    f.write("-"*90 + "\n")
    
    for r in all_results:
        f.write(f"{r['z_mm']:<10.0f} {r['r_flame_temp']:<25.2f} "
                f"{r['r_flame_grad']:<25.2f}\n")
    
    f.write("\n" + "="*90 + "\n")

print("✓ Master summary created: Case1/case1_master_summary.txt")

# =============================================================================
# BUG TESTING PLOTS
# =============================================================================
print("\n" + "="*70)
print("Creating diagnostic plots for bug testing...")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Get colors for each axial position
colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

# Plot 1: Temperature vs Radius
ax1.set_xlabel('Radius [mm]', fontsize=12)
ax1.set_ylabel('Temperature [K]', fontsize=12)
ax1.set_title('Temperature Profiles at Different Axial Positions', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Species vs Radius
ax2.set_xlabel('Radius [mm]', fontsize=12)
ax2.set_ylabel('Mass Fraction', fontsize=12)
ax2.set_title('Species Profiles at Different Axial Positions', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Read each processed CSV and plot
for idx, (sheet_name, z_position) in enumerate(axial_positions.items()):
    if sheet_name not in xls.sheet_names:
        continue
    
    case_name = f'case1_z{int(z_position*1000):03d}mm'
    df_plot = pd.read_csv(os.path.join(output_folder, f'{case_name}_processed.csv'))
    
    r_mm = df_plot['radius'] * 1000  # Convert to mm
    label = f'z={int(z_position*1000)}mm'
    
    # Plot temperature
    ax1.plot(r_mm, df_plot['temp'], color=colors[idx], linewidth=2, label=label)
    
    # Plot species (only for the last position to avoid clutter)
    if idx == len(axial_positions) - 1:
        ax2.plot(r_mm, df_plot['h2'], 'b-', linewidth=2, label='H₂')
        ax2.plot(r_mm, df_plot['o2'], 'r-', linewidth=2, label='O₂')
        ax2.plot(r_mm, df_plot['h2o'], 'g-', linewidth=2, label='H₂O')

ax1.legend(fontsize=10)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'bug_test_plots.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Diagnostic plots saved: {output_folder}/bug_test_plots.png")

# done! 
print("\n" + "="*70)
print("✅ PROCESSING COMPLETE")
print("="*70)
print(f"\nAll files saved to folder: {output_folder}/")
print(f"  • {len(all_results)} × _processed.csv (full data with calculations)")
print(f"  • {len(all_results)} × _summary.txt (individual summaries)")
print(f"  • 1 × case1_master_summary.txt (comparison across all z-positions)")
print(f"  • 1 × bug_test_plots.png (diagnostic plots)")
print("="*70)