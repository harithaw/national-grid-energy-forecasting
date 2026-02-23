import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/merged-generation-profile-2017-2026.csv', na_values=['Data N/A'])
solar_cols = [
    'Solar (Energy in GWh/day)',
    'Rooftop Solar (Est.) (Energy in GWh/day)',
    'Solar (Estimated) (Energy in GWh/day)',
]
mini_cols = [
    'Mini Hydro (Telemetered) (Energy in GWh/day)',
    'Mini Hydro (Estimated) (Energy in GWh/day)',
]
df['Date'] = pd.to_datetime(df['Date (GMT+5:30)'])
df.set_index('Date', inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print("=== Solar column temporal coverage ===")
for c in solar_cols:
    s = df[c].dropna()
    first = s.index.min().date() if len(s) else 'all NaN'
    last  = s.index.max().date() if len(s) else '-'
    pct   = 100 * len(s) / len(df)
    print(f"  {c[:42]:43s}  first={first}  last={last}  n={len(s):,}  ({pct:.1f}%)")

print()
print("=== Mini Hydro column temporal coverage ===")
for c in mini_cols:
    s = df[c].dropna()
    first = s.index.min().date() if len(s) else 'all NaN'
    last  = s.index.max().date() if len(s) else '-'
    pct   = 100 * len(s) / len(df)
    print(f"  {c[:46]:47s}  first={first}  last={last}  n={len(s):,}  ({pct:.1f}%)")

print()
print("=== Where multiple Solar columns are simultaneously populated (both > 0) ===")
s_raw = df[solar_cols[0]].fillna(0)
s_est = df[solar_cols[2]].fillna(0)
s_rf  = df[solar_cols[1]].fillna(0)

both_raw_est  = (s_raw > 0) & (s_est > 0)
both_raw_rf   = (s_raw > 0) & (s_rf  > 0)
both_est_rf   = (s_est > 0) & (s_rf  > 0)
all_three     = (s_raw > 0) & (s_est > 0) & (s_rf > 0)
print(f"  Solar & Solar_Estimated both > 0            : {both_raw_est.sum():,} days")
print(f"  Solar & Rooftop_Solar   both > 0            : {both_raw_rf.sum():,} days")
print(f"  Solar_Estimated & Rooftop_Solar both > 0    : {both_est_rf.sum():,} days")
print(f"  All three > 0                               : {all_three.sum():,} days")
print()
print("  Sample rows where Solar AND Solar_Estimated are both > 0:")
sub = df.loc[both_raw_est, solar_cols].head(8)
print(sub.to_string() if len(sub) else "  (none)")

print()
print("=== Additive vs alternative relationship check for Solar ===")
# If Solar_Estimated is a replacement for Solar, their values should be correlated
# If Rooftop_Solar is ADDITIVE it should represent a different physical source
overlap_mask = (s_raw > 0) & (s_est > 0)
if overlap_mask.sum() > 0:
    corr_raw_est = df.loc[overlap_mask, solar_cols[0]].corr(df.loc[overlap_mask, solar_cols[2]])
    diff_raw_est = (df.loc[overlap_mask, solar_cols[0]] - df.loc[overlap_mask, solar_cols[2]]).abs().mean()
    print(f"  Corr(Solar, Solar_Estimated) on overlap days = {corr_raw_est:.4f}")
    print(f"  Mean |Solar - Solar_Estimated|               = {diff_raw_est:.4f} GWh")

# For Rooftop Solar - is it separate from the main Solar figure?
overlap_mask2 = (s_raw > 0) & (s_rf > 0)
if overlap_mask2.sum() > 0:
    corr_rf = df.loc[overlap_mask2, solar_cols[0]].corr(df.loc[overlap_mask2, solar_cols[1]])
    diff_rf = (df.loc[overlap_mask2, solar_cols[0]] - df.loc[overlap_mask2, solar_cols[1]]).abs().mean()
    print(f"  Corr(Solar, Rooftop_Solar) on overlap days   = {corr_rf:.4f}")
    print(f"  Mean |Solar - Rooftop_Solar|                 = {diff_rf:.4f} GWh")

print()
print("=== Mini Hydro overlap analysis ===")
mh_tel = df[mini_cols[0]].fillna(0)
mh_est = df[mini_cols[1]].fillna(0)
both_mh = (mh_tel > 0) & (mh_est > 0)
print(f"  Both Telemetered & Estimated > 0: {both_mh.sum():,} days")
print()
print("  Sample rows where both > 0:")
sub_mh = df.loc[both_mh, mini_cols].head(8)
print(sub_mh.to_string() if len(sub_mh) else "  (none)")
if both_mh.sum() > 0:
    corr_mh = mh_tel[both_mh].corr(mh_est[both_mh])
    diff_mh = (mh_tel[both_mh] - mh_est[both_mh]).abs().mean()
    print(f"\n  Corr(Telemetered, Estimated) on overlap days = {corr_mh:.4f}")
    print(f"  Mean |Telemetered - Estimated|               = {diff_mh:.4f} GWh")

print()
print("=== Year-by-year Solar contributions (annual mean GWh/day) ===")
df['Solar_sum']   = s_raw + s_rf + s_est   # naive sum
df['Solar_Best']  = np.where(s_est > 0, s_est, s_raw)  # current v2 logic
# Proposed: Solar_Total = Solar_Best + Rooftop (if Rooftop is additive)
df['Solar_Total'] = df['Solar_Best'] + s_rf

yearly = df[['Solar_sum','Solar_Best','Solar_Total']].resample('YE').mean().round(3)
yearly.index = yearly.index.year
print(yearly.to_string())

print()
print("=== Year-by-year Mini Hydro contributions (annual mean GWh/day) ===")
df['MH_sum']  = mh_tel + mh_est
df['MH_Best'] = np.where(mh_est > 0, mh_est, mh_tel)  # current v2 logic
yearly_mh = df[['MH_sum','MH_Best']].resample('YE').mean().round(3)
yearly_mh.index = yearly_mh.index.year
print(yearly_mh.to_string())
