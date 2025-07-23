# Evaluation System Documentation

## Overview

The evaluation system in MCTS-LLM Alpha Mining Framework provides comprehensive multi-dimensional assessment of alpha factors. It combines real market data evaluation via Qlib with a sophisticated relative ranking mechanism.

## Architecture

```
┌─────────────────────┐
│ ComprehensiveEvaluator │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐  ┌────▼────┐
│  Qlib   │  │Relative │
│Evaluator│  │ Ranking │
└────┬────┘  └────┬────┘
     │            │
     ▼            ▼
┌─────────────────────┐
│  Dimension Metrics  │
│ • Effectiveness     │
│ • Stability         │
│ • Turnover         │
│ • Diversity        │
│ • Overfitting      │
└─────────────────────┘
```

## Evaluation Dimensions

### 1. Effectiveness (有效性)

**What it measures**: Predictive power and signal strength

**Metrics**:
- **IC (Information Coefficient)**: Correlation between factor values and future returns
- **ICIR (IC Information Ratio)**: IC_mean / IC_std

**Scoring**:
```python
# IC mapping to 0-10 score
IC < -0.2: 0
IC ∈ [-0.2, -0.05]: 0-2
IC ∈ [-0.05, 0]: 2-5
IC ∈ [0, 0.05]: 5-8
IC ∈ [0.05, 0.2]: 8-10
IC > 0.2: 10
```

**Threshold**: 3.0 (factors with score < 3.0 are not added to repository)

### 2. Stability (稳定性)

**What it measures**: Consistency of signal across time

**Metrics**:
- IC variance over time
- Performance consistency

**Scoring**:
```python
# ICIR mapping using tanh
score = 5 * (1 + np.tanh(icir / 2))
```

### 3. Turnover (换手率)

**What it measures**: Trading frequency (lower is better)

**Metrics**:
- Position change frequency
- Signal stability

**Scoring**:
```python
# Higher raw turnover → Higher score (less trading)
score = 10 * np.sqrt(turnover_stability)
```

### 4. Diversity (多样性)

**What it measures**: Uniqueness compared to existing factors

**Metrics**:
- Maximum absolute correlation with repository factors
- Feature overlap analysis

**Scoring**:
```python
# Lower correlation → Higher diversity score
diversity = 1 - max_abs_correlation
score = 10 * diversity
```

**Threshold**: 2.0 (minimum diversity required)

### 5. Overfitting (过拟合)

**What it measures**: Generalization ability

**Metrics**:
- In-sample vs out-of-sample performance gap
- Currently returns default value of 0.5

**Scoring**:
```python
score = 10 * (overfitting_metric ** 0.7)
```

## Relative Ranking Mechanism

The system implements the paper's relative ranking formula:

```
R_f^d = (1/N) * Σ I(score_f < score_i)
```

Where:
- `R_f^d`: Relative rank of factor f in dimension d
- `N`: Number of factors in repository
- `I()`: Indicator function (1 if true, 0 if false)

### Implementation Details

```python
def calculate_relative_rank(self, dimension, value, repo_values):
    if dimension == 'Turnover':
        # For Turnover, higher values are worse
        worse_count = sum(1 for v in repo_values if v > value)
    else:
        # For other dimensions, lower values are worse
        worse_count = sum(1 for v in repo_values if v < value)
    
    return worse_count / len(repo_values)
```

### Cold Start Handling

During initial iterations when repository is small:

1. **Phase 1** (repository < 3): Use base scores directly
2. **Phase 2** (repository ≥ 3): Apply relative ranking

## Evaluation Process

### 1. Formula Preparation
```python
# Clean and validate formula
formula = sanitize_formula(raw_formula)
formula = fix_missing_params(formula)
validated = validate_formula(formula)
```

### 2. Market Data Evaluation
```python
# Calculate factor values using Qlib
factor_expr = parse_formula_to_qlib(formula)
factor_data = D.features([factor_expr], start, end)

# Calculate returns
returns = calculate_future_returns()

# Compute metrics
ic = calculate_ic(factor_data, returns)
icir = calculate_icir(ic_series)
```

### 3. Score Calculation
```python
# Get raw metrics
raw_scores = {
    "IC": ic,
    "ICIR": icir,
    "Turnover": turnover,
    "Diversity": diversity,
    "Overfitting": overfitting
}

# Convert to normalized scores
scores = normalize_metrics(raw_scores)

# Apply relative ranking if applicable
if len(repository) >= 3:
    scores = apply_relative_ranking(scores, repository)
```

## Caching System

### Cache Key Generation
```python
cache_key = hashlib.md5(
    f"{formula}_{start_date}_{end_date}_{universe}".encode()
).hexdigest()
```

### Cache Structure
```python
{
    "cache_key": {
        "scores": {...},
        "raw_scores": {...},
        "factor_data": pd.DataFrame,
        "timestamp": datetime,
        "version": "1.0"
    }
}
```

### Cache Configuration
```yaml
evaluation:
  cache_size: 10000        # Maximum cache entries
  cache_ttl: 86400        # Time-to-live in seconds
  cache_dir: "./cache"    # Cache directory
```

## Alpha Repository

### Repository Entry Structure
```python
{
    "formula": "Rank(...)",
    "scores": {
        "Effectiveness": 7.5,
        "Stability": 8.2,
        "Turnover": 9.0,
        "Diversity": 8.5,
        "Overfitting": 5.0
    },
    "raw_scores": {...},
    "timestamp": datetime,
    "node_id": "node_123",
    "refinement_history": [...]
}
```

### Repository Management
- **Size limit**: 1000 factors
- **Admission criteria**: 
  - Effectiveness ≥ 3.0
  - Diversity ≥ 2.0
  - Overall ≥ 5.0
- **Maintenance**: Remove lowest scoring when full

## Performance Optimization

### 1. Batch Evaluation
```python
# Evaluate multiple formulas together
results = evaluator.evaluate_batch(formulas, n_jobs=4)
```

### 2. Parallel Processing
```python
# Use joblib for parallel IC calculation
with Parallel(n_jobs=n_jobs) as parallel:
    results = parallel(
        delayed(calculate_ic)(f) for f in formulas
    )
```

### 3. Vectorized Operations
```python
# Use pandas/numpy for efficient computation
ic_series = factor_data.corrwith(returns, axis=1)
```

## Error Handling

### Common Errors and Solutions

1. **Invalid Formula Syntax**
```python
try:
    ast.parse(formula)
except SyntaxError:
    return None, "Invalid formula syntax"
```

2. **Missing Market Data**
```python
if factor_data.isna().all():
    return None, "No valid data for evaluation"
```

3. **Evaluation Timeout**
```python
with timeout(300):  # 5 minute timeout
    result = evaluate_formula(formula)
```

## Configuration Options

```yaml
evaluation:
  # Qlib settings
  universe: "csi300"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  
  # Performance settings
  batch_size: 1000
  n_jobs: 4
  cache_size: 10000
  
  # Score thresholds
  effectiveness_threshold: 3.0
  diversity_threshold: 2.0
  overall_threshold: 5.0
  
  # Display settings
  show_raw_scores: true
  verbose: true
```

## Best Practices

1. **Data Quality**
   - Ensure Qlib data is up-to-date
   - Use sufficient historical data (3+ years)
   - Handle missing data appropriately

2. **Evaluation Efficiency**
   - Enable caching for repeated evaluations
   - Use appropriate batch sizes
   - Monitor memory usage

3. **Score Interpretation**
   - Consider both normalized and raw scores
   - Look for consistent performance across dimensions
   - Monitor relative ranking changes

4. **Repository Management**
   - Regularly review repository quality
   - Ensure diversity is maintained
   - Track performance over time

## Troubleshooting

### Issue: Low Effectiveness Scores
- Check data quality and date ranges
- Verify formula syntax and operators
- Ensure proper market universe selection

### Issue: High Turnover
- Consider adding smoothing operators (Mean, EMA)
- Increase window parameters
- Check for noisy input features

### Issue: Low Diversity
- Review repository for similar patterns
- Encourage structural formula changes
- Adjust diversity threshold if needed

## API Reference

### Main Evaluation Function
```python
def evaluate_formula(
    formula: str,
    repo_factors: List[Dict],
    node: Optional[MCTSNode] = None
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float]]:
    """
    Evaluate a formula across all dimensions.
    
    Returns:
        - scores: Normalized scores (0-10)
        - factor_data: Raw factor values
        - raw_scores: Original metric values
    """
```

### Batch Evaluation
```python
def evaluate_batch(
    formulas: List[str],
    repo_factors: List[Dict],
    n_jobs: int = 4
) -> List[Tuple[Dict, pd.DataFrame, Dict]]:
    """Evaluate multiple formulas in parallel."""
```

## See Also

- [Raw Metrics Display](raw_metrics_display.md)
- [Configuration Guide](configuration.md)
- [Architecture Overview](architecture.md)