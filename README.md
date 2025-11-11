#  Does More Data = Better Predictions?

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

**Debunking the myth: "More data always means better predictions"**

*A practical experiment exploring the surprising relationship between sample size and prediction accuracy*

[🚀 Quick Start](#-quick-start) • [📊 Key Findings](#-key-findings) • [💡 The Surprise](#-the-surprise)

</div>

---

## 🤔 The Question

> **"What happens to prediction errors when we increase our dataset from 100 to 10,000 samples?"**

Most people think: *"Obviously, more data = lower errors!"*  
**Reality:** 🤯 **It's more nuanced than that...**

---

## 🔬 The Experiment

This repository contains a **single, focused experiment** that challenges common assumptions about data and machine learning:

- 📈 Test **5 different dataset sizes**: 100 → 500 → 1,000 → 5,000 → 10,000
- 🔄 Run **100 repetitions** for statistical reliability
- 📉 Track **two key metrics**: Average error & Variability
- 🎨 Visualize the **surprising results**

**Total computational experiments: 500 models trained!**

---

## 💡 The Surprise

### What Most People Expect:
```
More Data → Lower Prediction Error ✓
```

### What Actually Happens:
```
More Data → Same Average Error 😮
More Data → Much Lower Variability ✓✓✓
```

**The Insight:**  
Sample size affects **PRECISION** (consistency), not **ACCURACY** (correctness)!

---

## 📊 Key Findings

| Sample Size | Avg RMSE | Std Dev | What This Means |
|-------------|----------|---------|-----------------|
| 100 | 2.66 | 0.27 | 🎲 High variability |
| 10,000 | 2.60 | 0.02 | 🎯 Very consistent |
| **Change** | **-2%** | **-91%** | **Precision ↑, Accuracy ≈** |

### 📉 Visual Proof

The code generates two compelling visualizations:

1. **Average RMSE vs Sample Size** - Flat line (accuracy unchanged)
2. **Std Dev vs Sample Size** - Dramatic drop (precision improved)

---

## 🎯 Why This Matters

### For Data Scientists:
- ❌ Don't expect miracles from just collecting more data
- ✅ Focus on **feature quality** and **model choice** for better predictions
- ✅ Use larger samples for **reliable estimates**, not lower errors

### For Business Leaders:
- 💰 **More data ≠ better predictions** (saves $$ on unnecessary data collection)
- 🎯 **Better features > more rows** (invest wisely)
- 📊 Larger samples = more **confidence**, not better **accuracy**

### For Students:
- 🧠 Practical demonstration of **bias-variance tradeoff**
- 📚 Real-world application of statistical theory
- 💻 Clean, reproducible code for learning

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Run the Experiment
```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/does-more-data-mean-better-predictions.git
cd does-more-data-mean-better-predictions

# Run the experiment (takes ~2-3 minutes)
python experiment.py
```

### Expected Output
```
Processing n = 100...     ✓ Done (RMSE=2.66)
Processing n = 500...     ✓ Done (RMSE=2.61)
Processing n = 1000...    ✓ Done (RMSE=2.60)
Processing n = 5000...    ✓ Done (RMSE=2.60)
Processing n = 10000...   ✓ Done (RMSE=2.60)

✓ Plot saved: analysis.png

ANSWER: Option 1 ✓
"On average, the RMSE does not change much as n gets larger,
while the variability of RMSE does decrease."
```

---

## 📖 The Story Behind This

This experiment is from **Exercise 31.2** of Rafael Irizarry's excellent book:  
[*Introduction to Data Science*](https://rafalab.dfci.harvard.edu/dsbook/)

**The specific question (Multiple choice):**

> *"Describe what you observe with the RMSE as the size of the dataset becomes larger."*

**Options:**
1. ✅ Average RMSE stays constant, variability decreases
2. ❌ RMSE decreases (law of large numbers)
3. ❌ Need even larger n to see effects
4. ❌ RMSE is not random

**This code proves Option 1 is correct through computational experiment!**

---

## 🧪 The Science

### What We're Testing
- **Bivariate Normal Data**: x and y with correlation ρ = 0.5
- **Linear Regression**: Simple y ~ x model
- **50/50 Split**: Train on half, test on half
- **RMSE Metric**: Root Mean Squared Error

### Why This Design?
- ✅ **Simple enough** to understand
- ✅ **Complex enough** to reveal insights
- ✅ **Reproducible** with fixed random seed
- ✅ **Statistically rigorous** (100 reps)

---



---

## 🎓 Learning Outcomes

After running this experiment, you'll understand:

1. **Bias-Variance Tradeoff** (practical demonstration)
2. **Precision vs Accuracy** (not the same thing!)
3. **Sample Size Effects** (when it helps, when it doesn't)
4. **Reproducible Research** (clean code, clear results)

---

## 💬 Discussion Questions

Thinking about running this? Consider:

1. What would happen if we changed ρ from 0.5 to 0.95?
2. Why does variability decrease but average stay constant?
3. When SHOULD we collect more data?
4. What's more valuable: more rows or better features?

**Hint:** The answers are all in the code output! 🤓



## 📚 Further Reading

Want to dive deeper? Check out:

- 📖 [Introduction to Data Science](https://rafalab.dfci.harvard.edu/dsbook/) - The source material
- 📊 [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) - The theory
- 🔬 [Statistical Power](https://en.wikipedia.org/wiki/Power_of_a_test) - Why n matters for inference

---

## 📜 License

MIT License - Feel free to use this for education, teaching, or learning!

---

## 🙏 Acknowledgments

- **Rafael Irizarry** for the excellent textbook and exercises
- **The Data Science Community** for making education accessible
- **You** for being curious about data! 🌟

---

<div align="center">

### ⭐ Star this repo if it changed your perspective on data!

**Remember:** More data makes you more *confident*, not necessarily more *correct*.

[🔝 Back to Top](#-does-more-data--better-predictions)

</div>

---

## 📊 Preview

### What You'll See:

**Console Output:**
```
======================================================================
ANALISIS: Apa yang terjadi dengan RMSE?
======================================================================

1. AVERAGE RMSE:
   n=100:    2.6582
   n=10000:  2.5991
   Change:   -2.22% ← HAMPIR TIDAK BERUBAH

2. STD DEV RMSE:
   n=100:    0.2669
   n=10000:  0.0247
   Change:   -90.7% ← TURUN DRASTIS
```

**Visual Output:**  
Two side-by-side plots showing:
1. Flat average RMSE (doesn't change with n)
2. Decreasing std dev (improves dramatically with n)

---

<div align="center">

**Built with ❤️ for data science education**

*Because understanding > memorizing*

</div>
