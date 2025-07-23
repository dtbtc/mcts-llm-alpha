# MCTS Algorithm Overview

This page explains how Monte Carlo Tree Search (MCTS) is adapted for alpha factor mining.

## Core Concepts

### What is MCTS?

Monte Carlo Tree Search is a heuristic search algorithm that builds a search tree incrementally and uses random sampling to evaluate positions. In our context:

- **Nodes**: Represent alpha formulas
- **Edges**: Represent refinements along specific dimensions
- **Value**: Multi-dimensional quality scores

### The Four Phases

```
┌─────────────┐
│  Selection  │ ──► Use UCT to traverse tree
└─────────────┘
       ↓
┌─────────────┐
│  Expansion  │ ──► Create new formula via LLM
└─────────────┘
       ↓
┌─────────────┐
│ Evaluation  │ ──► Calculate 5-dimensional scores
└─────────────┘
       ↓
┌─────────────┐
│Backpropagation│ ──► Update tree statistics
└─────────────┘
```

## Algorithm Details

### 1. Selection Phase

Uses Upper Confidence Bound for Trees (UCT):

```python
UCT(node) = Q(node) + c * sqrt(ln(N_parent) / N_node)
```

Where:
- `Q(node)`: Average value of node (normalized to [0,1])
- `c`: Exploration constant (default: 1.0)
- `N_parent`: Parent visit count
- `N_node`: Node visit count

### 2. Expansion Phase

Two types of expansion:

#### Initial Expansion
```python
1. Generate alpha portrait (high-level description)
2. Convert to symbolic formula: f(w1, w2, w3)
3. Generate 3 candidate parameter sets
4. Evaluate and select best parameters
```

#### Refinement Expansion
```python
1. Select dimension via softmax: P(d) = exp(budget_d / T)
2. Use Few-shot examples from repository
3. Generate refined formula maintaining structure
4. Virtual action for internal nodes
```

### 3. Evaluation Phase

Five dimensions scored 0-10:

1. **Effectiveness** (IC, ICIR)
   ```python
   IC = correlation(factor_values, future_returns)
   ICIR = mean(IC) / std(IC)
   ```

2. **Stability** (max drawdown, variance)
   ```python
   stability = 10 * (1 - max_drawdown) * consistency_ratio
   ```

3. **Turnover** (trading frequency)
   ```python
   turnover = daily_position_changes / total_positions
   score = 10 * (1 - min(turnover / threshold, 1))
   ```

4. **Diversity** (correlation with existing)
   ```python
   max_corr = max(correlations_with_repository)
   diversity = 10 * (1 - abs(max_corr))
   ```

5. **Overfitting** (train/test gap)
   ```python
   gap = abs(train_score - test_score)
   overfitting = 10 * (1 - min(gap / threshold, 1))
   ```

### 4. Backpropagation Phase

Update node statistics:
```python
node.visit_count += 1
node.value_sum += overall_score
node.value = value_sum / visit_count
```

## Key Innovations

### Virtual Action Mechanism

Allows expansion of internal nodes:
```python
if is_internal_node and should_expand:
    # Create virtual expansion
    new_child = expand_with_virtual_action(node)
    # Continue search from new child
```

### Relative Ranking

Dynamic scoring based on repository:
```python
R_f^d = (1/N) * sum(I(score_f < score_i))
final_score = 10 * R_f^d
```

### Dimension-Specific Refinement

Each dimension has tailored prompts:
- **Effectiveness**: "Enhance signal strength"
- **Stability**: "Improve robustness"
- **Turnover**: "Reduce trading frequency"
- **Diversity**: "Increase uniqueness"
- **Overfitting**: "Improve generalization"

## Configuration Parameters

```yaml
mcts:
  max_iterations: 50
  exploration_constant: 1.0
  dimension_temperature: 10.0
  max_expansions_per_dimension: 2
  virtual_action_probability: 0.3
```

## Performance Considerations

1. **Caching**: Evaluation results are cached
2. **Parallel Evaluation**: Multiple formulas evaluated concurrently
3. **Early Stopping**: Poor branches pruned early
4. **Checkpoint**: Save/resume long searches

## Example Search Tree

```
Root: Momentum Factor
├── Effectiveness: Add volume confirmation
│   ├── Stability: Smooth with MA
│   └── Diversity: Add rank transform
├── Stability: Use longer window
│   └── Turnover: Add threshold
└── Turnover: Reduce frequency
    ├── Effectiveness: Maintain signal
    └── Virtual: Hybrid approach
```

## References

- Original MCTS paper: Browne et al. (2012)
- UCT formula: Kocsis & Szepesvári (2006)
- Application to finance: [Paper reference]

For implementation details, see [Architecture](Architecture) page.