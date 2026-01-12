import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

# Excel file
excel_file = 'case1_data.xlsx'
output_folder = 'Case1_Results'

SI_target = 0.3  # Adjusted target for H2O-based SI
axial_positions = {
    '2': 0.05,    # 50mm
    '3': 0.15,    # 150mm
    '4': 0.25,    # 250mm
    '5': 0.35,    # 350mm
}

os.makedirs(output_folder, exist_ok=True)

# Extract data
xls = pd.ExcelFile(excel_file)
results = []

for sheet_name, z_mm in axial_positions.items():
    if sheet_name not in xls.sheet_names:
        continue
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['radius'] < 0.1].copy()
    
    # Filter to reacting regions only
    df = df[df['h2o'] > 0.001].copy()
    
    if len(df) < 5:
        print(f"⚠️ Too few reacting points at z={z_mm*1000}mm")
        continue
    
    # H2O-based Stratification Index
    mean_h2o = df['h2o'].mean()
    std_h2o = df['h2o'].std()
    SI_h2o = std_h2o / mean_h2o if mean_h2o > 1e-6 else 0
    
    # H2O radial gradient
    df['dh2o_dr'] = np.gradient(df['h2o'], df['radius'])
    mean_h2o_grad = np.abs(df['dh2o_dr']).mean()
    
    # Temperature uniformity
    temp_uniformity = 1 - (df['temp'].std() / df['temp'].mean())
    
    # Turbulence metrics
    TKE = df['tke'].mean()
    TI = df['ti'].mean()
    
    results.append({
        'z_mm': z_mm * 1000,
        'SI_h2o': SI_h2o,
        'mean_h2o': mean_h2o,
        'mean_h2o_grad': mean_h2o_grad,
        'TKE': TKE,
        'TI': TI,
        'T_uniformity': temp_uniformity,
        'T_max': df['temp'].max(),
        'n_points': len(df)
    })

df_results = pd.DataFrame(results)

# Inlet conditions
TI_inlet = df_results.iloc[0]['TI']
TKE_inlet = df_results.iloc[0]['TKE']

print(f"\nInlet Conditions (z={df_results.iloc[0]['z_mm']:.0f}mm)")
print(f"   Initial SI_h2o    = {df_results.iloc[0]['SI_h2o']:.3f}")
print(f"   Inlet TKE         = {TKE_inlet:.1f} m²/s²")
print(f"   Inlet TI          = {TI_inlet:.2f}%")

print(f"\nAxial Results:")
print("="*90)
print(f"{'z [mm]':>8} {'SI_h2o':>10} {'mean H2O':>12} {'dH2O/dr':>12} {'T_unif':>10} {'TI [%]':>8} {'TKE':>8}")
print("-"*90)
for idx, row in df_results.iterrows():
    print(f"{row['z_mm']:>8.0f} {row['SI_h2o']:>10.3f} {row['mean_h2o']:>12.4f} "
          f"{row['mean_h2o_grad']:>12.2f} {row['T_uniformity']:>10.3f} "
          f"{row['TI']:>8.2f} {row['TKE']:>8.2f}")
print("="*90)

print(f"\nData Check:")
print(f"   SI_h2o values: {df_results['SI_h2o'].tolist()}")

# Check monotonicity
is_monotonic = all(df_results['SI_h2o'].iloc[i] >= df_results['SI_h2o'].iloc[i+1] 
                   for i in range(len(df_results)-1))
if is_monotonic:
    print(f"   ✓ All data points show monotonic SI_h2o decrease")
else:
    print(f"   ⚠️ Non-monotonic behavior detected")

# Gradients
df_results['dSI_dz'] = np.gradient(df_results['SI_h2o'], df_results['z_mm'])
df_results['dTKE_dz'] = np.gradient(df_results['TKE'], df_results['z_mm'])

# Mixing regime
transition_idx = df_results['TKE'].idxmax()
z_transition = df_results.loc[transition_idx, 'z_mm']

print("\nMixing Regime Detection:")
print("="*80)
print(f"   Near-field → Far-field transition at z ≈ {z_transition:.0f} mm")
print("="*80)

# ========== ADD THIS BLOCK ==========
print("\nRegime Characterization:")
print("="*80)

# Split data by regime
near_data = df_results[df_results['z_mm'] <= z_transition]
far_data = df_results[df_results['z_mm'] > z_transition]

# Near-field trends
near_SI_change = near_data['SI_h2o'].iloc[-1] - near_data['SI_h2o'].iloc[0]
near_TKE_change = near_data['TKE'].iloc[-1] - near_data['TKE'].iloc[0]

print(f"NEAR-FIELD (50-{z_transition:.0f}mm):")
print(f"  TKE change: {near_TKE_change:+.2f} m²/s² ({'production' if near_TKE_change > 0 else 'dissipation'})")
print(f"  SI_h2o change: {near_SI_change:+.3f} ({'homogenizing' if near_SI_change < 0 else 'stratifying'})")
print(f"  dSI/dz average: {near_data['dSI_dz'].mean():.5f}")
print(f"  Mechanism: Jet turbulence drives rapid product mixing")

# Far-field trends
far_SI_change = far_data['SI_h2o'].iloc[-1] - far_data['SI_h2o'].iloc[0]
far_TKE_change = far_data['TKE'].iloc[-1] - far_data['TKE'].iloc[0]

print(f"\nFAR-FIELD ({z_transition:.0f}-350mm):")
print(f"  TKE change: {far_TKE_change:+.2f} m²/s² ({'production' if far_TKE_change > 0 else 'dissipation'})")
print(f"  SI_h2o change: {far_SI_change:+.3f} ({'homogenizing' if far_SI_change < 0 else 'stratifying'})")
print(f"  dSI/dz average: {far_data['dSI_dz'].mean():.5f}")
print(f"  Mechanism: Products spread into non-uniform thermal field")

print(f"\nTRANSITION POINT (z={z_transition:.0f}mm):")
print(f"  TKE at maximum: {df_results.loc[transition_idx, 'TKE']:.2f} m²/s²")
print(f"  SI_h2o at minimum: {df_results.loc[transition_idx, 'SI_h2o']:.3f}")
print(f"  Temperature uniformity: {df_results.loc[transition_idx, 'T_uniformity']:.3f}")
print("="*80)



# Exponential decay fit
z = df_results['z_mm'].values
SI = df_results['SI_h2o'].values

def exp_decay(z, SI_0, k, SI_inf):
    return SI_inf + (SI_0 - SI_inf) * np.exp(-k * z)

try:
    params, _ = curve_fit(exp_decay, z, SI, 
                         p0=[SI[0], 0.005, SI[-1]*0.5],
                         maxfev=10000)
    SI_0, k, SI_inf = params
    
    if k > 0 and 0 <= SI_inf < SI_0:
        fit_quality = True
        print(f"\n✓ Exponential fit successful:")
        print(f"   SI_h2o(z) = {SI_inf:.3f} + {SI_0-SI_inf:.3f}×exp(-{k:.5f}×z)")
        print(f"   Decay constant k = {k:.5f} mm⁻¹")
        print(f"   Characteristic length ℓ = {1/k:.1f} mm")
    else:
        print(f"   ⚠️ Fit gave non-physical values (k={k:.5f}, SI_inf={SI_inf:.3f})")
        fit_quality = False
        k = None
except Exception as e:
    print(f"   ⚠️ Curve fitting failed: {str(e)}")
    fit_quality = False
    k = None

# Mixing length
if df_results['SI_h2o'].min() < SI_target:
    interp = interp1d(df_results['SI_h2o'][::-1], df_results['z_mm'][::-1], 
                     kind='linear', fill_value='extrapolate')
    L_mix = float(interp(SI_target))
    print(f"\n   Mixing length (SI_h2o={SI_target}) = {L_mix:.1f} mm")
else:
    L_mix = None
    print(f"\n   SI_h2o does not reach target {SI_target}")

# Piecewise fits
near_df = df_results.loc[:transition_idx].copy()
far_df  = df_results.loc[transition_idx:].copy()

print("\nPiecewise Mixing Decay Constants:")
print("="*80)
if k_near:
    print(f"Near-field: k = {k_near:.5f} mm⁻¹, ℓ = {1/k_near:.1f} mm")
if k_far:
    print(f"Far-field : k = {k_far:.5f} mm⁻¹, ℓ = {1/k_far:.1f} mm")
if k_near and k_far:
    print(f"k_near / k_far ≈ {k_near/k_far:.1f}")

# ========== ADD THIS BLOCK ==========
print("\nRegime-Specific Correlations:")
print("="*80)

# Near-field: TKE vs SI correlation
from scipy.stats import pearsonr
if len(near_data) >= 2:
    r_near, p_near = pearsonr(near_data['TKE'], near_data['SI_h2o'])
    print(f"Near-field TKE-SI correlation: r = {r_near:.3f} (p={p_near:.3f})")
    print(f"  {'Strong negative' if r_near < -0.5 else 'Weak'} correlation: High TKE → Low SI")

# Far-field: Gradient persistence
if len(far_data) >= 2:
    r_far, p_far = pearsonr(far_data['TKE'], far_data['mean_h2o_grad'])
    print(f"\nFar-field TKE-Gradient correlation: r = {r_far:.3f} (p={p_far:.3f})")
    print(f"  Gradients {'persist' if abs(r_far) < 0.5 else 'decay'} despite TKE decrease")
print("="*80)
# ========== END BLOCK ==========


k_near = None
if len(near_df) >= 2:
    try:
        z_near = near_df['z_mm'].iloc[:2].values
        SI_near = near_df['SI_h2o'].iloc[:2].values
        SI0_near = SI_near[0]
        SIinf_near = SI_near[1]
        
        def exp_decay_2pt(z, k):
            return SIinf_near + (SI0_near - SIinf_near) * np.exp(-k * (z - z_near[0]))
        
        popt, _ = curve_fit(exp_decay_2pt, z_near, SI_near, p0=[0.01])
        k_near = popt[0]
    except:
        pass

k_far = None
if len(far_df) >= 2:
    try:
        z_far_shifted = far_df['z_mm'] - far_df['z_mm'].iloc[0]
        params_far, _ = curve_fit(exp_decay, z_far_shifted, far_df['SI_h2o'],
                                  p0=[far_df['SI_h2o'].iloc[0], 0.003, far_df['SI_h2o'].iloc[-1]],
                                  maxfev=10000)
        SI0_far, k_far, SIinf_far = params_far
    except:
        pass

print("\nPiecewise Mixing Decay Constants:")
print("="*80)
if k_near:
    print(f"Near-field: k = {k_near:.5f} mm⁻¹, ℓ = {1/k_near:.1f} mm")
if k_far:
    print(f"Far-field : k = {k_far:.5f} mm⁻¹, ℓ = {1/k_far:.1f} mm")
if k_near and k_far:
    print(f"k_near / k_far ≈ {k_near/k_far:.1f}")

# Turbulent mixing efficiency
df_results['SI_reduction'] = df_results.iloc[0]['SI_h2o'] - df_results['SI_h2o']
df_results['integral_TKE'] = np.cumsum(df_results['TKE'] * np.gradient(df_results['z_mm']))

valid_idx = df_results['integral_TKE'] > 0
if valid_idx.sum() > 2:
    efficiency = df_results.loc[valid_idx, 'SI_reduction'] / df_results.loc[valid_idx, 'integral_TKE']
    avg_efficiency = efficiency.mean()
    print(f"\nTurbulent Mixing Efficiency:")
    print(f"   ΔSI_h2o per (TKE×length) = {avg_efficiency:.6f} (m²/s²·mm)⁻¹")

# Plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SI_h2o Decay
ax1.plot(df_results['z_mm'], df_results['SI_h2o'], 'ro-', linewidth=2, markersize=10, label='Data')
ax1.axvline(z_transition, color='purple', linestyle='--', linewidth=2, alpha=0.6, label='Near/Far transition')

if fit_quality:
    z_fit = np.linspace(z.min(), z.max(), 100)
    SI_fit = exp_decay(z_fit, SI_0, k, SI_inf)
    ax1.plot(z_fit, SI_fit, 'b--', linewidth=2, label=f'Global fit (k={k:.5f})')

if L_mix:
    ax1.axhline(SI_target, color='green', linestyle='--', linewidth=2, label=f'Target SI={SI_target}')
    ax1.axvline(L_mix, color='green', linestyle=':', linewidth=2, alpha=0.5)

ax1.set_xlabel('Axial Distance [mm]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Product Stratification Index (H₂O)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Product Homogenization', fontsize=12, fontweight='bold')
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


# Plot 3: Regime-Specific SI Evolution
ax3.scatter(near_data['TKE'], near_data['SI_h2o'], s=150, c='red', 
           marker='o', label='Near-field', edgecolors='black', linewidths=2)
ax3.scatter(far_data['TKE'], far_data['SI_h2o'], s=150, c='blue',
           marker='s', label='Far-field', edgecolors='black', linewidths=2)
ax3.set_xlabel('Turbulent Kinetic Energy [m²/s²]', fontsize=11, fontweight='bold')
ax3.set_ylabel('SI_h2o', fontsize=11, fontweight='bold')
ax3.set_title('(c) Regime-Dependent Turbulence-Stratification Coupling', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Add arrows showing trends
if len(near_data) >= 2:
    ax3.annotate('', xy=(near_data['TKE'].iloc[-1], near_data['SI_h2o'].iloc[-1]),
                xytext=(near_data['TKE'].iloc[0], near_data['SI_h2o'].iloc[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
if len(far_data) >= 2:
    ax3.annotate('', xy=(far_data['TKE'].iloc[-1], far_data['SI_h2o'].iloc[-1]),
                xytext=(far_data['TKE'].iloc[0], far_data['SI_h2o'].iloc[0]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))


# Plot 4: Temperature Uniformity
ax4.plot(df_results['z_mm'], df_results['T_uniformity'], 'go-', linewidth=2, markersize=10)
ax4.set_xlabel('Axial Distance [mm]', fontsize=11, fontweight='bold')
ax4.set_ylabel('Temperature Uniformity', fontsize=11, fontweight='bold')
ax4.set_title('(d) Thermal Homogenization', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3)

# ========== ADD THIS BLOCK BEFORE plt.savefig() ==========
# Add regime annotation to Plot 1
ax1.axvspan(df_results['z_mm'].min(), z_transition, alpha=0.1, color='red', label='Near-field')
ax1.axvspan(z_transition, df_results['z_mm'].max(), alpha=0.1, color='blue', label='Far-field')
ax1.legend()
# ========== END BLOCK ==========

plt.tight_layout()
plt.savefig(f'{output_folder}/Design_Summary_H2O.png', dpi=300)
plt.show()

plt.tight_layout()
plt.savefig(f'{output_folder}/Design_Summary_H2O.png', dpi=300)
plt.show()

# Save results
summary = {
    'Inlet_TI_%': TI_inlet,
    'Inlet_TKE_m2s2': TKE_inlet,
    'Initial_SI_h2o': df_results.iloc[0]['SI_h2o'],
    'Final_SI_h2o': df_results.iloc[-1]['SI_h2o'],
    'Mixing_Length_mm': L_mix if L_mix else 'N/A',
    'Decay_Constant_mm-1': k if fit_quality else 'N/A',
    'k_near': k_near if k_near else 'N/A',
    'k_far': k_far if k_far else 'N/A'
}

summary_df = pd.DataFrame([summary])
summary_df.to_excel(f'{output_folder}/Design_Metrics_H2O.xlsx', index=False)
df_results.to_excel(f'{output_folder}/Detailed_Results_H2O.xlsx', index=False)

print(f"\n✓ Saved: {output_folder}/Design_Summary_H2O.png")
print(f"✓ Saved: {output_folder}/Design_Metrics_H2O.xlsx")
