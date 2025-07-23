# Configuration Guide

## Overview

The MCTS-LLM Alpha Mining Framework uses a hierarchical configuration system that supports YAML files, environment variables, and command-line arguments.

## Configuration Hierarchy

Configuration sources are loaded in the following priority order (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Custom YAML configuration file
4. Default configuration (`config/default.yaml`)

## Configuration Structure

### Main Configuration File

```yaml
# config/default.yaml
mcts:
  max_iterations: 50              # Total MCTS search iterations
  exploration_constant: 1.0       # UCT exploration vs exploitation
  budget_increment: 0.1           # Budget increase per iteration
  max_depth: 10                   # Maximum tree depth
  max_nodes: 5000                 # Maximum nodes in tree
  checkpoint_freq: 10             # Checkpoint save frequency
  dimension_temperature: 10.0     # Softmax temperature for dimension selection
  effectiveness_threshold: 3.0    # Minimum effectiveness score
  diversity_threshold: 2.0        # Minimum diversity score
  overall_threshold: 5.0          # Minimum overall score
  max_expansions_per_dimension: 2 # Max expansions per dimension per node

llm:
  model: "gpt-4o-mini"           # OpenAI model to use
  max_retries: 3                 # API retry attempts
  retry_delay: 1.0               # Delay between retries
  temperature: 0.7               # Default generation temperature

data:
  provider: "qlib"               # Data provider type
  qlib_provider_uri: null        # Qlib data directory (uses env var if null)
  universe: "csi300"             # Stock universe
  start_date: "2020-01-01"       # Evaluation start date
  end_date: "2023-12-31"         # Evaluation end date

evaluation:
  batch_size: 1000               # Evaluation batch size
  cache_size: 10000              # Evaluation cache size
  n_jobs: 4                      # Parallel jobs for evaluation
  verbose: true                  # Verbose output
  diversity_threshold: 2.0       # Diversity threshold
  overall_threshold: 5.0         # Overall score threshold

paths:
  results_dir: "./results"       # Results directory
  checkpoint_dir: "./checkpoints" # Checkpoint directory
  log_dir: "./logs"              # Log directory
  cache_dir: null                # Cache directory (uses temp if null)

logging:
  level: "INFO"                  # Log level
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_logging: true             # Enable file logging
  console_logging: true          # Enable console logging
```

## Environment Variables

The following environment variables are supported:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key        # OpenAI API key

# Data paths
QLIB_PROVIDER_URI=/path/to/qlib/data  # Qlib data directory

# Optional overrides
LOG_LEVEL=DEBUG                        # Override log level
RESULTS_DIR=./my_results               # Override results directory
CHECKPOINT_DIR=./my_checkpoints        # Override checkpoint directory
CACHE_DIR=/tmp/mcts_cache             # Override cache directory

# Performance tuning
EVALUATION_N_JOBS=8                    # Number of parallel evaluation jobs
EVALUATION_BATCH_SIZE=500              # Evaluation batch size
```

## Command-Line Arguments

### Basic Usage

```bash
# Run with default configuration
mcts-llm-alpha search

# Specify iterations
mcts-llm-alpha search --iterations 100

# Use custom configuration
mcts-llm-alpha search --config my_config.yaml

# Override specific parameters
mcts-llm-alpha search --iterations 100 --exploration-constant 1.5
```

### Available Arguments

```bash
# Search command arguments
--iterations          # Override max_iterations
--config             # Path to custom config file
--seed-formula       # Initial seed formula
--exploration-constant # UCT exploration parameter
--checkpoint-freq    # Checkpoint frequency
--verbose           # Enable verbose output

# Resume command arguments
--checkpoint        # Path to checkpoint file
--iterations       # Additional iterations to run
```

## Configuration Examples

### 1. Quick Exploration Config

```yaml
# config/quick_exploration.yaml
mcts:
  max_iterations: 20
  exploration_constant: 2.0  # More exploration
  checkpoint_freq: 5
  dimension_temperature: 20.0  # More random dimension selection

evaluation:
  batch_size: 500  # Smaller batches for speed
  verbose: false   # Less output
```

### 2. Production Search Config

```yaml
# config/production.yaml
mcts:
  max_iterations: 500
  exploration_constant: 0.5  # More exploitation
  checkpoint_freq: 50
  max_nodes: 10000
  effectiveness_threshold: 4.0  # Higher standards
  diversity_threshold: 3.0
  overall_threshold: 6.0

llm:
  model: "gpt-4"  # Better model
  temperature: 0.5  # More focused generation

evaluation:
  batch_size: 2000
  n_jobs: 16  # More parallel processing
```

### 3. Debug Config

```yaml
# config/debug.yaml
mcts:
  max_iterations: 5
  checkpoint_freq: 1

logging:
  level: "DEBUG"

evaluation:
  verbose: true
  show_intermediate: true
```

## Scoring Configuration

The scoring system is configured in `scoring_config.py`:

```python
# Dimension thresholds
EFFECTIVENESS_THRESHOLD = 3.0
DIVERSITY_THRESHOLD = 2.0
OVERALL_THRESHOLD = 5.0

# Score ranges
MIN_SCORE = 0.0
MAX_SCORE = 10.0

# UCT normalization
UCT_MIN = 0.0
UCT_MAX = 1.0

# Repository settings
MAX_REPOSITORY_SIZE = 1000
MIN_REPOSITORY_SIZE = 10
```

## Performance Tuning

### 1. Memory Usage

```yaml
evaluation:
  batch_size: 500      # Reduce if OOM
  cache_size: 5000     # Reduce cache size
  
mcts:
  max_nodes: 2000      # Limit tree size
```

### 2. Speed Optimization

```yaml
evaluation:
  n_jobs: -1           # Use all CPU cores
  batch_size: 2000     # Larger batches
  
llm:
  model: "gpt-3.5-turbo"  # Faster model
```

### 3. Quality Optimization

```yaml
mcts:
  exploration_constant: 0.5
  max_iterations: 1000
  effectiveness_threshold: 4.0
  
llm:
  model: "gpt-4"
  temperature: 0.3
```

## Custom Extensions

### Adding New Configuration Parameters

1. Update the configuration schema:
```python
# config/schema.py
class MCTSConfig(BaseModel):
    max_iterations: int = 50
    my_new_param: float = 1.0  # Add new parameter
```

2. Use in code:
```python
config = load_config()
value = config.mcts.my_new_param
```

### Configuration Validation

The framework uses Pydantic for validation:

```python
from mcts_llm_alpha.config import load_config, MCTSConfig

# Load and validate
config = load_config("my_config.yaml")

# Programmatic modification
config.mcts.max_iterations = 100
config.validate()  # Ensures consistency
```

## Best Practices

1. **Start with defaults**: The default configuration is well-tuned
2. **Use environment variables for secrets**: Never commit API keys
3. **Create task-specific configs**: Different configs for exploration vs production
4. **Monitor resource usage**: Adjust batch_size and n_jobs based on hardware
5. **Save configurations**: Keep configs with results for reproducibility

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Set environment variable: `export OPENAI_API_KEY=sk-...`
   - Or create `.env` file with the key

2. **"Qlib data not found"**
   - Set `QLIB_PROVIDER_URI` to correct path
   - Download data: `python -m qlib.run.get_data qlib_data`

3. **Out of Memory**
   - Reduce `batch_size` and `cache_size`
   - Reduce `max_nodes` in MCTS config

4. **Slow evaluation**
   - Increase `n_jobs` for more parallelism
   - Use larger `batch_size`
   - Enable caching with proper `cache_dir`