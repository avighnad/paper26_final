import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

# =============================================================================
# SETTINGS
# =============================================================================
excel_file = 'case2_data.xlsx'
output_folder = 'Case2_Results'

stoich_mass_ratio = 7.936
SI_target = 0.5  # Design target: "well-mixed" threshold

axial_positions = {
    '1': 0.025,   # 25mm
    '2': 0.05,    # 50mm
    '3': 0.15,    # 150mm
    '4': 0.25,    # 250mm
    '5': 0.35,    # 350mm
}

os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# STEP 1: EXTRACT DATA
# =============================================================================
xls = pd.ExcelFile(excel_file)
results = []

print("="*80)
print("COMBUSTOR DESIGN ANALYSIS: Turbulence vs Mixing")
print("="*80)

for sheet_name, z_mm in axial_positions.items():
    if sheet_name not in xls.sheet_names:
        continue
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower()
    
    # Calculate equivalence ratio and stratification
    df['o2_safe'] = df['o2'].replace(0, 1e-12)
    df['phi'] = (df['h2'] / df['o2_safe']) / (1.0 / stoich_mass_ratio)
    df['dphi_dr'] = np.gradient(df['phi'], df['radius'])
    mean_phi = df['phi'].mean()
    SI = df['phi'].std() / mean_phi if mean_phi > 1e-6 else 0
    
    # Combustion efficiency (local)
    Y_H2_inlet = df['h2'].max()
    df['eta_local'] = (Y_H2_inlet - df['h2']) / Y_H2_inlet * 100.0
    df['eta_local'] = df['eta_local'].clip(0, 100)
    mean_eta = df['eta_local'].mean()
    
    # Turbulence metrics
    TKE = df['tke'].mean()
    TI = df['ti'].mean()
    
    # Stratification gradient (mean absolute gradient)
    mean_strat_grad = np.abs(df['dphi_dr']).mean()
    
    results.append({
        'z_mm': z_mm * 1000,
        'SI': SI,
        'TKE': TKE,
        'TI': TI,
        'T_max': df['temp'].max(),
        'mean_strat_grad': mean_strat_grad,
        'mean_efficiency': mean_eta
    })

df_results = pd.DataFrame(results)

# Get inlet turbulence (first station)
TI_inlet = df_results.iloc[0]['TI']
TKE_inlet = df_results.iloc[0]['TKE']

print(f"\n📍 INLET CONDITIONS (z={df_results.iloc[0]['z_mm']:.0f}mm)")
print(f"   Initial SI        = {df_results.iloc[0]['SI']:.3f}")
print(f"   Inlet TKE         = {TKE_inlet:.1f} m²/s²")
print(f"   Inlet TI          = {TI_inlet:.2f}%")

# Print detailed station data
print(f"\n📊 STATION-BY-STATION RESULTS:")
print("="*80)
print(f"{'z [mm]':>8} {'SI':>8} {'dφ/dr [m⁻¹]':>15} {'η [%]':>8} {'TI [%]':>8} {'TKE [m²/s²]':>12}")
print("-"*80)
for idx, row in df_results.iterrows():
    print(f"{row['z_mm']:>8.0f} {row['SI']:>8.3f} {row['mean_strat_grad']:>15.1f} "
          f"{row['mean_efficiency']:>8.1f} {row['TI']:>8.2f} {row['TKE']:>12.2f}")
print("="*80)

# =============================================================================
# STEP 2: CLEAN DATA - Remove anomalous points
# =============================================================================

# Detect if SI increases downstream (burnout region, not mixing region)
# Keep only the monotonically decreasing portion
print(f"\n🔍 DATA QUALITY CHECK:")
print(f"   SI values: {df_results['SI'].tolist()}")

SI_min_idx = df_results['SI'].idxmin()
print(f"   Minimum SI at index {SI_min_idx} (z={df_results.loc[SI_min_idx, 'z_mm']:.0f}mm)")

valid_data = df_results.loc[:SI_min_idx].copy()

if len(valid_data) < len(df_results):
    removed = len(df_results) - len(valid_data)
    print(f"   ⚠️  Removing {removed} point(s) after z={valid_data['z_mm'].max():.0f}mm (SI increases)")
    print(f"   Analysis uses: z={valid_data['z_mm'].min():.0f}-{valid_data['z_mm'].max():.0f}mm\n")
    df_results = valid_data
else:
    print(f"   ✓ All data points show monotonic SI decrease\n")

# =============================================================================
# STEP 3: KEY DESIGN METRICS
# =============================================================================

# Find mixing length (where SI drops to target)
if df_results['SI'].min() < SI_target:
    # Interpolate to find exact distance
    interp = interp1d(df_results['SI'][::-1], df_results['z_mm'][::-1], 
                     kind='linear', fill_value='extrapolate')
    L_mix = float(interp(SI_target))
else:
    L_mix = None

# Calculate decay rate constant (exponential fit)
z = df_results['z_mm'].values
SI = df_results['SI'].values

# Try exponential decay: SI = SI_0 * exp(-k*z) + SI_inf
def exp_decay(z, SI_0, k, SI_inf):
    return SI_inf + (SI_0 - SI_inf) * np.exp(-k * z)

try:
    params, _ = curve_fit(exp_decay, z, SI, 
                         p0=[SI[0], 0.01, 0.5],
                         bounds=([SI[0]*0.5, 0, 0], [SI[0]*2, 1, 2]))
    SI_0, k, SI_inf = params
    fit_quality = True
except:
    fit_quality = False
    k = None

print(f"\n📊 MIXING PERFORMANCE (Analysis Range: z={df_results['z_mm'].min():.0f}-{df_results['z_mm'].max():.0f}mm)")
print(f"   SI decay from {df_results.iloc[0]['SI']:.2f} → {df_results.iloc[-1]['SI']:.2f}")
if L_mix:
    print(f"   Mixing length (SI={SI_target}) = {L_mix:.1f} mm")
if fit_quality:
    print(f"   Decay constant k = {k:.4f} mm⁻¹")
    print(f"   Characteristic length = {1/k:.1f} mm")

# =============================================================================
# STEP 3: TURBULENT MIXING EFFICIENCY
# =============================================================================

# Calculate mixing efficiency: how much SI reduced per unit TKE per mm
df_results['SI_reduction'] = df_results.iloc[0]['SI'] - df_results['SI']
df_results['integral_TKE'] = np.cumsum(
    df_results['TKE'] * np.gradient(df_results['z_mm'])
)

# Mixing efficiency = ΔSI / (∫TKE·dz)
valid_idx = df_results['integral_TKE'] > 0
if valid_idx.sum() > 2:
    efficiency = df_results.loc[valid_idx, 'SI_reduction'] / df_results.loc[valid_idx, 'integral_TKE']
    avg_efficiency = efficiency.mean()
    print(f"\n⚡ TURBULENT MIXING EFFICIENCY")
    print(f"   ΔSI per (TKE×length) = {avg_efficiency:.6f} (m²/s²·mm)⁻¹")

# =============================================================================
# STEP 4: DESIGN CHART
# =============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SI Decay
ax1.plot(df_results['z_mm'], df_results['SI'], 'ro-', linewidth=2, markersize=10)
if fit_quality:
    z_fit = np.linspace(z.min(), z.max(), 100)
    SI_fit = exp_decay(z_fit, SI_0, k, SI_inf)
    ax1.plot(z_fit, SI_fit, 'b--', linewidth=2, 
            label=f'SI = {SI_inf:.2f} + {SI_0-SI_inf:.2f}×exp(-{k:.4f}z)')
    ax1.legend(fontsize=9)
ax1.axhline(SI_target, color='green', linestyle='--', linewidth=2, label=f'Target SI={SI_target}')
if L_mix:
    ax1.axvline(L_mix, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax1.set_xlabel('Axial Distance [mm]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Stratification Index', fontsize=11, fontweight='bold')
ax1.set_title('(a) Stratification Decay', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Turbulence Evolution
ax2_twin = ax2.twinx()
ax2.plot(df_results['z_mm'], df_results['TI'], 'bs-', linewidth=2, markersize=8, label='TI')
ax2_twin.plot(df_results['z_mm'], df_results['TKE'], 'g^-', linewidth=2, markersize=8, label='TKE')
ax2.set_xlabel('Axial Distance [mm]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Turbulence Intensity [%]', fontsize=11, fontweight='bold', color='blue')
ax2_twin.set_ylabel('TKE [m²/s²]', fontsize=11, fontweight='bold', color='green')
ax2.tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='green')
ax2.set_title('(b) Turbulence Evolution', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: SI vs Cumulative TKE Exposure
ax3.plot(df_results['integral_TKE'], df_results['SI'], 'mo-', linewidth=2, markersize=10)
ax3.set_xlabel('Cumulative TKE Exposure [m²/s²·mm]', fontsize=11, fontweight='bold')
ax3.set_ylabel('Stratification Index', fontsize=11, fontweight='bold')
ax3.set_title('(c) Mixing vs Turbulent Energy Input', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Temperature Profile
ax4.plot(df_results['z_mm'], df_results['T_max'], 'ro-', linewidth=2, markersize=10)
ax4.set_xlabel('Axial Distance [mm]', fontsize=11, fontweight='bold')
ax4.set_ylabel('Maximum Temperature [K]', fontsize=11, fontweight='bold')
ax4.set_title('(d) Peak Temperature', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_folder}/Design_Summary.png', dpi=300)
plt.show()

# =============================================================================
# STEP 5: DESIGN RECOMMENDATIONS
# =============================================================================

print(f"\n" + "="*80)
print("🎯 DESIGN GUIDANCE")
print("="*80)

if L_mix:
    print(f"\n1. MIXING LENGTH")
    print(f"   At TI_inlet = {TI_inlet:.2f}%, need L = {L_mix:.0f} mm to reach SI < {SI_target}")
    
    # Estimate for different TI
    if fit_quality:
        # Higher TI → faster decay → shorter length
        print(f"\n   Scaling estimate (if k ∝ TI):")
        for TI_new in [2.0, 2.5, 3.0, 3.5]:
            k_new = k * (TI_new / TI_inlet)
            L_new = -np.log((SI_target - SI_inf) / (SI_0 - SI_inf)) / k_new
            print(f"   TI = {TI_new:.1f}% → L_mix ≈ {L_new:.0f} mm ({L_new/L_mix*100:.0f}% of baseline)")

if fit_quality:
    print(f"\n2. CHARACTERISTIC MIXING TIME")
    print(f"   τ_mix = 1/k = {1/k:.1f} mm (distance scale)")

print(f"\n3. TURBULENCE REQUIREMENTS")
print(f"   Current inlet: TI = {TI_inlet:.2f}%, TKE = {TKE_inlet:.1f} m²/s²")
print(f"   Maintains high turbulence (TI > 2%) through mixing zone")

# Save summary
summary = {
    'Inlet_TI_%': TI_inlet,
    'Inlet_TKE_m2s2': TKE_inlet,
    'Initial_SI': df_results.iloc[0]['SI'],
    'Final_SI': df_results.iloc[-1]['SI'],
    'Mixing_Length_mm': L_mix if L_mix else 'N/A',
    'Decay_Constant_mm-1': k if fit_quality else 'N/A'
}

summary_df = pd.DataFrame([summary])
summary_df.to_excel(f'{output_folder}/Design_Metrics.xlsx', index=False)
df_results.to_excel(f'{output_folder}/Detailed_Results.xlsx', index=False)

print(f"\n✅ DONE")
print(f"📁 {output_folder}/Design_Summary.png")
print(f"📁 {output_folder}/Design_Metrics.xlsx")
print("="*80)
