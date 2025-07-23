# Raw Evaluation Metrics Display

## Overview

The MCTS-LLM Alpha system now supports displaying both normalized scores (0-10 range) and raw evaluation metric values, allowing users to gain deeper insights into the actual performance of alpha factors.

## Raw Metrics Displayed

The system displays the following raw evaluation metrics:

### 1. **IC (Information Coefficient)**
- Correlation between factor values and future returns
- Typical range: -0.05 to 0.05
- Higher is better (positive values indicate positive correlation)
- Interpretation:
  - IC > 0.03: Excellent predictive power
  - IC ∈ [0.01, 0.03]: Good predictive power
  - IC ∈ [0, 0.01]: Weak predictive power
  - IC < 0: Negative correlation (may need signal reversal)

### 2. **ICIR (Information Coefficient Information Ratio)**
- Stability indicator of IC, calculated as mean(IC) / std(IC)
- Typical range: -2 to 2
- Higher is better (indicates stable signal)
- Interpretation:
  - ICIR > 1.5: Very stable signal
  - ICIR ∈ [0.5, 1.5]: Moderately stable signal
  - ICIR < 0.5: Unstable signal

### 3. **Turnover**
- Signal stability (between 0-1)
- Higher is better (high value indicates low turnover)
- Represents the fraction of positions that remain unchanged
- Important for transaction cost considerations

### 4. **Diversity**
- Difference from existing factors (between 0-1)
- Higher is better (high value indicates more uniqueness)
- Based on maximum absolute correlation with repository factors

### 5. **Overfitting**
- Consistency between in-sample and out-of-sample performance (between 0-1)
- Higher is better (high value indicates low overfitting)
- Currently returns default value of 0.5

## Output Format Example

```
📊 Initial Formula Information:
Formula: Rank((($close - Ref($close, 10)) / Ref($close, 10)) * Mean($volume, 10)), 5)
Normalized Scores: Effectiveness=7.50, Stability=8.20, Turnover=9.00, Diversity=8.50, Overfitting=5.00
Overall Score: 7.64
Raw Metrics: IC=0.0234, ICIR=1.8567, Turnover=0.8234, Diversity=0.7890, Overfitting=0.5000

Top 5 Repository Alpha Factors:

[1] Rank((($close - Mean($close, 20)) / Mean($close, 20)), 10) * ($volume / Mean($volume, 15))
    Normalized Scores:
      Effectiveness: 7.50
      Stability: 8.20
      Turnover: 9.00
      Diversity: 8.50
      Overfitting: 5.00
      Overall: 7.64
    Raw Metric Values:
      IC: 0.0234
      ICIR: 1.8567
      Turnover: 0.8234
      Diversity: 0.7890
      Overfitting: 0.5000
```

## Score Mapping Explanation

The system uses the following mappings to convert raw metrics to 0-10 scores:

### IC → Effectiveness Score
```python
# Piecewise linear mapping
IC < -0.2: 0
IC ∈ [-0.2, -0.05]: 0-2
IC ∈ [-0.05, 0]: 2-5
IC ∈ [0, 0.05]: 5-8
IC ∈ [0.05, 0.2]: 8-10
IC > 0.2: 10
```

### ICIR → Stability Score
```python
# Tanh function for smooth mapping
score = 5 * (1 + np.tanh(icir / 2))
# Maps ICIR ∈ [-5, 5] to scores ∈ [0, 10]
```

### Turnover → Turnover Score
```python
# Square root function for better discrimination
score = 10 * np.sqrt(turnover)
# Emphasizes differences in low turnover region
```

### Diversity → Diversity Score
```python
# Linear mapping
score = 10 * diversity
# Direct proportion from [0, 1] to [0, 10]
```

### Overfitting → Overfitting Score
```python
# Power function mapping
score = 10 * (overfitting ** 0.7)
# Makes high scores harder to achieve
```

## Usage Recommendations

### 1. **Comprehensive Evaluation**
- Don't rely solely on normalized scores
- Raw values provide more nuanced information
- Both IC and ICIR should be high for a good factor

### 2. **Trade-off Considerations**
- High IC with low ICIR may indicate unstable signals
- Low turnover (high turnover score) helps reduce transaction costs
- Balance effectiveness with stability and turnover

### 3. **Repository Comparison**
- Compare raw metrics with repository averages
- Look for factors that improve on multiple dimensions
- Consider the relative ranking within the repository

## Configuration Options

Adjust display options in the configuration file:

```yaml
display:
  show_raw_scores: true  # Whether to show raw scores
  decimal_places: 4      # Decimal places for raw scores
  
evaluation:
  verbose: true          # Enable detailed output
  show_intermediate: true # Show intermediate calculations
```

## Technical Implementation

### Data Structure
```python
# Evaluation result includes both normalized and raw scores
{
    "scores": {
        "Effectiveness": 7.5,
        "Stability": 8.2,
        "Turnover": 9.0,
        "Diversity": 8.5,
        "Overfitting": 5.0
    },
    "raw_scores": {
        "IC": 0.0234,
        "ICIR": 1.8567,
        "Turnover": 0.8234,
        "Diversity": 0.7890,
        "Overfitting": 0.5000
    }
}
```

### Cache Compatibility
- Cached results include raw metric values
- Old cache entries without raw values are automatically updated
- Cache key includes evaluation parameters

## Notes and Limitations

1. **Data Requirements**
   - Raw metrics are only available with real data evaluation
   - Simulation mode does not produce meaningful raw metrics

2. **Metric Stability**
   - IC and ICIR may vary with market conditions
   - Use sufficient historical data for reliable estimates
   - Consider different time periods for robustness

3. **Future Enhancements**
   - Additional metrics (Sharpe ratio, maximum drawdown)
   - Time-series visualization of metrics
   - Market regime-specific analysis

## See Also

- [Evaluation System Documentation](evaluation_system.md)
- [Configuration Guide](configuration.md)
- [Architecture Overview](architecture.md)