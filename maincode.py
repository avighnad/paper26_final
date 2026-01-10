import pandas as pd
import numpy as np

# =============================================================================
# SETTINGS - Change these for your analysis
# =============================================================================
data_file = 'case1_data.xlsx'

# Which sheets to process and their axial positions (in meters)
axial_positions = {
    '1': 0.1,
    '2': 0.2,
    '3': 0.3,
    '4': 0.4,
    '5': 0.5,
    '6': 0.6
}

# Constants for hydrogen combustion
STOICH_RATIO = 7.936  # kg O2 per kg H2
FLAME_TEMP_THRESHOLD = 1800  # K, temperature to define reaction zone

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
    
    # ---------------------------------------------------------
    # STEP 1: Load and prepare data
    # ---------------------------------------------------------
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    df = df[['radius', 'temp', 'h2', 'h2o', 'o2']].copy()
    
    # ---------------------------------------------------------
    # STEP 2: Calculate equivalence ratio
    # ---------------------------------------------------------
    # Avoid division by zero in fuel-rich regions
    df['o2_safe'] = df['o2'].replace(0, 1e-12)
    
    # φ = (H2/O2)_actual / (H2/O2)_stoich
    df['phi'] = (df['h2'] / df['o2_safe']) / (1.0 / STOICH_RATIO)
    
    # Calculate how phi changes with radius
    df['dphi_dr'] = np.gradient(df['phi'], df['radius'])
    
    # ---------------------------------------------------------
    # STEP 3: Stratification metrics
    # ---------------------------------------------------------
    mean_phi = df['phi'].mean()
    std_phi = df['phi'].std()
    
    # Stratification Index = std/mean (coefficient of variation)
    SI = std_phi / mean_phi
    
    # RMS gradient (emphasizes large gradients)
    rms_gradient = np.sqrt(np.mean(df['dphi_dr']**2))
    
    # Find the steepest gradient location
    max_gradient = abs(df['dphi_dr']).max()
    max_gradient_location = df.loc[abs(df['dphi_dr']).idxmax(), 'radius'] * 1000  # mm
    
    # ---------------------------------------------------------
    # STEP 4: Calculate local combustion efficiency
    # ---------------------------------------------------------
    # Inlet H2 is the maximum value (before combustion)
    H2_inlet = df['h2'].max()
    
    # Efficiency = how much H2 has been consumed
    df['eta_local'] = ((H2_inlet - df['h2']) / H2_inlet * 100).clip(0, 100)
    
    mean_efficiency = df['eta_local'].mean()
    
    # ---------------------------------------------------------
    # STEP 5: Find flame location
    # ---------------------------------------------------------
    # Temperature gradient shows where the flame front is
    df['dT_dr'] = np.gradient(df['temp'], df['radius'])
    
    # Method 1: Where temperature is highest
    max_temp = df['temp'].max()
    r_at_max_temp = df.loc[df['temp'].idxmax(), 'radius'] * 1000  # mm
    
    # Method 2: Where temperature gradient is steepest
    r_at_max_temp_grad = df.loc[abs(df['dT_dr']).idxmax(), 'radius'] * 1000  # mm
    
    # Method 3: Flame thickness (radial extent of hot region)
    hot_region = df[df['temp'] > FLAME_TEMP_THRESHOLD]
    if len(hot_region) > 0:
        flame_inner = hot_region['radius'].min() * 1000
        flame_outer = hot_region['radius'].max() * 1000
        flame_thickness = flame_outer - flame_inner
    else:
        flame_thickness = 0.0
    
    # ---------------------------------------------------------
    # STEP 6: Save processed data
    # ---------------------------------------------------------
    case_name = f'case1_z{int(z_position*1000):03d}mm'
    df.to_csv(f'{case_name}_processed.csv', index=False)
    
    # ---------------------------------------------------------
    # STEP 7: Write summary text file
    # ---------------------------------------------------------
    with open(f'{case_name}_summary.txt', 'w') as f:
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
        f.write(f"Flame location (max dT/dr)   : {r_at_max_temp_grad:.2f} mm\n")
        f.write(f"Flame thickness (T > 1800K)  : {flame_thickness:.2f} mm\n\n")
        
        f.write("="*60 + "\n")
    
    # ---------------------------------------------------------
    # STEP 8: Store results for comparison
    # ---------------------------------------------------------
    all_results.append({
        'z_mm': z_position * 1000,
        'mean_phi': mean_phi,
        'SI': SI,
        'rms_grad': rms_gradient,
        'max_grad': max_gradient,
        'mean_eta': mean_efficiency,
        'max_temp': max_temp,
        'r_flame_temp': r_at_max_temp,
        'r_flame_grad': r_at_max_temp_grad,
        'flame_thickness': flame_thickness
    })
    
    print(f"   ✓ Mean φ = {mean_phi:.3f}, SI = {SI:.3f}, η = {mean_efficiency:.1f}%")

# =============================================================================
# CREATE MASTER SUMMARY FILE
# =============================================================================
print("\n" + "="*70)
print("Creating master summary file...")

with open('case1_master_summary.txt', 'w') as f:
    f.write("="*90 + "\n")
    f.write("MASTER SUMMARY - ALL AXIAL POSITIONS\n")
    f.write("="*90 + "\n\n")
    
    # Table 1: Main metrics at each location
    f.write("MAIN METRICS\n")
    f.write("-"*90 + "\n")
    f.write(f"{'z [mm]':<10} {'Mean φ':<10} {'SI':<10} {'RMS ∇φ':<12} "
            f"{'Max ∇φ':<12} {'η [%]':<10} {'T_max [K]':<12}\n")
    f.write("-"*90 + "\n")
    
    for r in all_results:
        f.write(f"{r['z_mm']:<10.0f} {r['mean_phi']:<10.3f} {r['SI']:<10.3f} "
                f"{r['rms_grad']:<12.1f} {r['max_grad']:<12.1f} "
                f"{r['mean_eta']:<10.1f} {r['max_temp']:<12.1f}\n")
    
    # Table 2: Evolution rates (how fast things change downstream)
    f.write("\n\nEVOLUTION RATES (per 100mm downstream)\n")
    f.write("-"*90 + "\n")
    f.write(f"{'z [mm]':<10} {'ΔSI/Δz':<12} {'Δ(RMS∇φ)/Δz':<15} "
            f"{'Δη/Δz [%]':<15} {'ΔT_max/Δz [K]':<15}\n")
    f.write("-"*90 + "\n")
    
    for i in range(1, len(all_results)):
        # Calculate change per 100mm
        dz = (all_results[i]['z_mm'] - all_results[i-1]['z_mm']) / 100
        
        dSI = (all_results[i]['SI'] - all_results[i-1]['SI']) / dz
        dRMS = (all_results[i]['rms_grad'] - all_results[i-1]['rms_grad']) / dz
        dEta = (all_results[i]['mean_eta'] - all_results[i-1]['mean_eta']) / dz
        dTemp = (all_results[i]['max_temp'] - all_results[i-1]['max_temp']) / dz
        
        f.write(f"{all_results[i]['z_mm']:<10.0f} {dSI:<12.4f} {dRMS:<15.2f} "
                f"{dEta:<15.2f} {dTemp:<15.1f}\n")
    
    # Table 3: Flame location evolution
    f.write("\n\nFLAME LOCATION\n")
    f.write("-"*90 + "\n")
    f.write(f"{'z [mm]':<10} {'r_flame (T_max) [mm]':<25} "
            f"{'r_flame (∇T) [mm]':<25} {'Thickness [mm]':<20}\n")
    f.write("-"*90 + "\n")
    
    for r in all_results:
        f.write(f"{r['z_mm']:<10.0f} {r['r_flame_temp']:<25.2f} "
                f"{r['r_flame_grad']:<25.2f} {r['flame_thickness']:<20.2f}\n")
    
    f.write("\n" + "="*90 + "\n")

print("✓ Master summary created: case1_master_summary.txt")

# =============================================================================
# DONE!
# =============================================================================
print("\n" + "="*70)
print("✅ PROCESSING COMPLETE")
print("="*70)
print(f"\nGenerated files:")
print(f"  • {len(all_results)} × _processed.csv (full data with calculations)")
print(f"  • {len(all_results)} × _summary.txt (individual summaries)")
print(f"  • 1 × case1_master_summary.txt (comparison across all z-positions)")
print("="*70)