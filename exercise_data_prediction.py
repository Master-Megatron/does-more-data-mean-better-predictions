"""
Exercise 31.2 - Nomor 3 SAJA
Jawab pertanyaan: Apa yang terjadi dengan RMSE ketika ukuran dataset lebih besar?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# =============================================================================
# FUNGSI HELPER
# =============================================================================
def generate_data(n, rho=0.5):
    """Generate bivariate normal data"""
    Sigma = 9 * np.array([[1.0, rho], [rho, 1.0]])
    mean = [69, 69]
    data = np.random.multivariate_normal(mean, Sigma, n)
    return pd.DataFrame(data, columns=['x', 'y'])

def compute_rmse(n, rho=0.5):
    """Single run: compute RMSE"""
    dat = generate_data(n, rho)
    train, test = train_test_split(dat, test_size=0.5, random_state=None)
    
    model = LinearRegression()
    model.fit(train[['x']], train['y'])
    y_hat = model.predict(test[['x']])
    
    return np.sqrt(mean_squared_error(test['y'], y_hat))

# =============================================================================
# EXERCISE 3: ANALISIS EFFECT OF SAMPLE SIZE
# =============================================================================
print("="*70)
print("EXERCISE 3: Effect of Sample Size pada RMSE")
print("="*70)

# Test different sample sizes
sample_sizes = [100, 500, 1000, 5000, 10000]
results = []

for n in sample_sizes:
    print(f"\nProcessing n = {n}...")
    
    # Run 100 repetitions
    rmse_list = [compute_rmse(n, rho=0.5) for _ in range(100)]
    
    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    
    results.append({
        'n': n,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse
    })
    
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Std Dev RMSE: {std_rmse:.4f}")

# Convert to DataFrame
df = pd.DataFrame(results)

print("\n" + "="*70)
print("TABEL HASIL")
print("="*70)
print(df.to_string(index=False))

# =============================================================================
# ANALISIS UNTUK MENJAWAB NOMOR 3
# =============================================================================
print("\n\n" + "="*70)
print("ANALISIS: Apa yang terjadi dengan RMSE?")
print("="*70)

# Perbandingan n=100 vs n=10000
rmse_100 = df.loc[df['n']==100, 'avg_rmse'].values[0]
rmse_10000 = df.loc[df['n']==10000, 'avg_rmse'].values[0]
std_100 = df.loc[df['n']==100, 'std_rmse'].values[0]
std_10000 = df.loc[df['n']==10000, 'std_rmse'].values[0]

change_rmse = ((rmse_10000 - rmse_100) / rmse_100) * 100
change_std = ((std_10000 - std_100) / std_100) * 100

print(f"\n1. AVERAGE RMSE:")
print(f"   n=100:    {rmse_100:.4f}")
print(f"   n=10000:  {rmse_10000:.4f}")
print(f"   Change:   {change_rmse:+.2f}% ← HAMPIR TIDAK BERUBAH")

print(f"\n2. STD DEV RMSE:")
print(f"   n=100:    {std_100:.4f}")
print(f"   n=10000:  {std_10000:.4f}")
print(f"   Change:   {change_std:+.1f}% ← TURUN DRASTIS")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Average RMSE
axes[0].plot(df['n'], df['avg_rmse'], marker='o', linewidth=2, 
             markersize=8, color='blue')
axes[0].set_xlabel('Sample Size (n)', fontsize=12)
axes[0].set_ylabel('Average RMSE', fontsize=12)
axes[0].set_title('Average RMSE vs Sample Size\n(Relatif Stabil)', 
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')

# Plot 2: Std Dev RMSE
axes[1].plot(df['n'], df['std_rmse'], marker='s', linewidth=2, 
             markersize=8, color='red')
axes[1].set_xlabel('Sample Size (n)', fontsize=12)
axes[1].set_ylabel('Std Dev of RMSE', fontsize=12)
axes[1].set_title('Variability of RMSE vs Sample Size\n(Turun Drastis)', 
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('exercise3_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved: exercise3_analysis.png")

# =============================================================================
# JAWABAN SOAL NOMOR 3
# =============================================================================
print("\n\n" + "="*70)
print("JAWABAN SOAL NOMOR 3")
print("="*70)

print("""
PERTANYAAN:
Describe what you observe with the RMSE as the size of the dataset 
becomes larger.

PILIHAN JAWABAN:
1. On average, the RMSE does not change much as n gets larger, 
   while the variability of RMSE does decrease. ✓

2. Because of the law of large numbers, the RMSE decreases: 
   more data, more precise estimates.

3. n = 10000 is not sufficiently large. To see a decrease in RMSE, 
   we need to make it larger.

4. The RMSE is not a random variable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JAWABAN: OPTION 1 ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PENJELASAN:

1. AVERAGE RMSE tidak berubah signifikan
   - Dari hasil: {:.4f} → {:.4f} (perubahan {:.2f}%)
   - Ini karena underlying relationship (ρ=0.5) tetap sama
   - Larger n tidak mengubah true model

2. VARIABILITY (Std Dev) menurun drastis
   - Dari hasil: {:.4f} → {:.4f} (turun {:.1f}%)
   - Ini karena estimates lebih stable dengan n besar
   - Less variability in coefficient estimates

3. KESIMPULAN:
   ✓ Larger n → More PRECISE estimates (lower variance)
   ✗ Larger n → NOT more ACCURATE predictions (RMSE tetap)
   
   Sample size helps with PRECISION, not ACCURACY!

MENGAPA BUKAN OPTION LAIN?

Option 2: SALAH
- Law of large numbers berlaku untuk estimates, bukan RMSE
- RMSE tidak turun karena underlying error tetap ada

Option 3: SALAH  
- n=10000 sudah cukup besar
- Std Dev sudah sangat kecil (0.02)
- RMSE tidak akan turun signifikan meski n lebih besar

Option 4: SALAH
- RMSE ADALAH random variable (depends on random split)
- Bukti: ada variability dalam 100 repetitions
""".format(rmse_100, rmse_10000, change_rmse, std_100, std_10000, abs(change_std)))

print("="*70)
print("✅ EXERCISE 3 SELESAI!")
print("="*70)
