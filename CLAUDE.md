# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-powered Monte Carlo Tree Search (MCTS) framework for formulaic alpha factor mining in quantitative finance. The system uses GPT-4 to generate and refine mathematical formulas that predict stock returns, evaluating them across multiple quality dimensions.

**Latest Update (2025-07-22)**: All 14 identified issues have been fixed to ensure full compliance with the original paper's algorithm. The system now correctly implements Virtual Action mechanism, uses 0-10 scoring range throughout, and includes symbolic parameter generation with Few-shot learning.

## Project Structure

The project has been refactored into a modular structure:
- **Original code**: `mcts_llm_full_v7_fixed.py` (single file implementation)
- **Refactored code**: `mcts-llm-alpha/` (modular package structure)

### Refactored Package Structure
```
mcts-llm-alpha/
├── src/mcts_llm_alpha/
│   ├── config/          # Configuration management
│   ├── llm/            # LLM integration (OpenAI client, prompts)
│   ├── mcts/           # Core MCTS algorithm
│   ├── formula/        # Formula processing pipeline
│   ├── evaluation/     # Multi-dimensional evaluation
│   │   └── metrics/    # Individual dimension evaluators
│   ├── data/           # Qlib data provider wrapper
│   └── cli.py          # Command-line interface
└── tests/              # Comprehensive test suite
```

## Key Commands

### Installation and Setup
```bash
# Activate conda environment
conda activate alphagen_dev

# Install refactored version in development mode
cd mcts-llm-alpha
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env to add OPENAI_API_KEY and QLIB_PROVIDER_URI
```

### Running MCTS Search
```bash
# Basic search (refactored version - recommended)
mcts-llm-alpha search --iterations 50

# With custom configuration
mcts-llm-alpha search --config config/experiment.yaml

# Resume from checkpoint
mcts-llm-alpha resume --checkpoint mcts_checkpoint_20240101_120000.pkl

# Original version
python mcts_llm_full_v7_fixed.py

# Jupyter notebook
jupyter notebook mcts_llm_full_v7_fixed.ipynb
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest -v --cov=mcts_llm_alpha --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_formula_utils.py

# Run specific test
pytest tests/test_mcts_core.py::test_node_selection

# Run fix validation tests
python tests/test_fixes_mcts_core.py      # MCTS core fixes
python tests/test_fixes_evaluation.py      # Evaluation system fixes
python tests/test_fixes_llm_integration.py # LLM integration fixes
python tests/test_fixes_fsa.py            # FSA fixes
python tests/test_fixes_config.py         # Configuration fixes
```

### Code Quality
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Architecture Overview

### Core Components

1. **MCTSNode** (`mcts/node.py` in refactored, lines 45-125 in original):
   - Tree node representing an alpha formula
   - Tracks formula refinement history and multi-dimensional scores (0-10 range)
   - Implements UCT-based selection logic with proper normalization
   - Manages expansion constraints (max 2 per dimension)
   - Removed complexity_budget mechanism (fixed in 1.4)

2. **MCTSSearch** (`mcts/search.py` in refactored, lines 1079-1345 in original):
   - Controls tree exploration/exploitation via softmax dimension selection (e_max=10)
   - Implements Virtual Action mechanism for internal node expansion
   - Manages alpha repository for successful formulas
   - Dynamic budget adjustment based on historical max score
   - Implements checkpoint saving for long runs

3. **Two-Step Generation Process**:
   - `generate_alpha_portrait()`: Creates high-level alpha description with pseudo-code
   - `generate_formula_from_portrait()`: Returns symbolic formula with parameter candidates
   - Symbolic parameters (w1, w2, t) with 3 candidate value sets
   - Few-shot example selection based on dimension

4. **Formula Processing Pipeline**:
   - `sanitize_formula()`: Cleans GPT output
   - `fix_missing_params()`: Comprehensive operator fixing and parameter adjustment
   - `validate_formula()`: Syntax and operator validation
   - `substitute_parameters()`: Replace symbolic parameters with concrete values

### Quality Dimensions

The system evaluates formulas across 5 dimensions:
- **Effectiveness**: Signal strength and prediction accuracy
- **Stability**: Robustness across market conditions  
- **Turnover**: Trading frequency (lower is better)
- **Diversity**: Uniqueness from existing factors
- **Overfitting**: Generalization capability

### External Dependencies

- **Qlib**: Quantitative library for factor calculation (data dir: `G:/workspace/qlib_bin/qlib_bin`)
- **OpenAI API**: Requires environment variable with API key
- **Standard**: numpy, pandas, ast, pickle, multiprocessing, networkx, pydantic

## Environment Setup

### Python Environment
- **Conda Environment**: `alphagen_dev`
- **Python Path**: `G:\ProgramData\anaconda3\envs\alphagen_dev\python.exe`
- **Activation**: `conda activate alphagen_dev`

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
QLIB_PROVIDER_URI=G:/workspace/qlib_bin/qlib_bin  # 必须设置！否则使用模拟数据
LOG_LEVEL=INFO
RESULTS_DIR=./results
CHECKPOINT_DIR=./checkpoints
```

### Dependencies
1. Install Qlib and configure data directory
2. Set OpenAI API key as environment variable
3. Ensure Windows multiprocessing compatibility (`freeze_support()`)
4. CSI300 universe data required for evaluation
5. Install package with dev dependencies: `pip install -e ".[dev]"`

### 重要：系统只支持真实数据
**本系统已完全移除模拟数据和模拟评估功能**。系统现在强制要求：

1. **必须安装Qlib**：`pip install qlib`
2. **必须有真实市场数据**：
   ```bash
   python -m qlib.run.get_data qlib_data --target_dir G:/workspace/qlib_bin/qlib_bin --region cn
   ```
3. **必须设置OpenAI API密钥**：系统需要真实的LLM进行公式生成
4. **必须设置QLIB_PROVIDER_URI**：指向正确的数据目录

如果缺少任何必需组件，系统会立即退出并提示错误信息。

### 已移除的模拟功能
- ~~MockDataProvider~~：已移除，系统只使用QlibDataProvider
- ~~测试模式生成器~~：已移除，必须使用真实LLM
- ~~模拟评估~~：已移除，只使用Qlib真实评估

## Development Guidelines

### Working with Formulas

1. **Qlib Operators**: The system supports specific operators like Mean, Std, Rank, Corr, etc. See `fix_missing_params()` for full operator mappings.

2. **Formula Constraints**:
   - Maximum 3 parameters per formula
   - Must use at least 2 different operators
   - Window parameters typically range 5-60

3. **Common Formula Patterns**:
   - Price/volume based: Uses $close, $open, $high, $low, $volume
   - Technical indicators: Moving averages, correlations, rankings
   - Cross-sectional: Rank, Zscore operators

### Configuration

The system supports configuration through:
- **YAML files**: Default at `src/mcts_llm_alpha/config/default.yaml`
- **Environment variables**: Via `.env` file
- **Command-line arguments**: Override any configuration

Key parameters:
- `max_iterations`: Total search iterations (default: 50)
- `exploration_constant`: UCT exploration vs exploitation (default: 1.0)
- `dimension_temperature`: Softmax temperature for dimension selection
- `checkpoint_freq`: How often to save checkpoints (default: 10)
- `effectiveness_threshold`: Minimum effectiveness score (default: 3.0)
- `diversity_threshold`: Minimum diversity score (default: 2.0)
- `overall_threshold`: Minimum overall score (default: 5.0)
- All thresholds use 0-10 scoring range

### Debugging Tips

1. **Formula Generation Issues**: Check `sanitize_formula()` and `fix_missing_params()` for operator conversion logic
2. **Evaluation Errors**: System uses Qlib for real evaluation (simulation mode removed)
3. **MCTS Behavior**: Monitor `dimension_temperature` and UCT exploration constant
4. **Scoring Range**: All scores should be in 0-10 range, UCT internally normalizes to 0-1
5. **Virtual Action**: Internal nodes can be expanded via virtual action mechanism
6. **FSA Patterns**: Only root genes (subtrees starting from fields) are mined

### Key Files

- `mcts_llm_full_v7_fixed.py`: Main implementation
- `mcts_llm_full_v7_fixed.ipynb`: Jupyter notebook version
- `mcts-llm-alpha/src/mcts_llm_alpha/cli.py`: CLI entry point
- `Artical/MTCS_LLM_Alpha.md`: English methodology documentation
- `Artical/alpha_mtcs_cn.md`: Chinese documentation

## Common Tasks

### Adding New Operators
Modify `fix_missing_params()` (or `formula/fixer.py` in refactored) to add operator mappings and parameter constraints.

### Adjusting Search Parameters
Key parameters in `MCTSSearch.__init__()`:
- `max_iterations`: Total search iterations
- `exploration_constant`: UCT exploration vs exploitation
- `dimension_temperature`: Softmax temperature for dimension selection

### Implementing Real Evaluation
Replace `eval_formula_simple()` with actual Qlib backtesting for production use. See `evaluation/qlib_evaluator.py` in refactored version.

### Running Single Tests
```bash
# Test specific functionality
pytest tests/test_formula_utils.py::test_sanitize_formula
pytest tests/test_mcts_core.py::test_node_expansion
```

### Checkpoint Management
- Checkpoints saved automatically every 10 iterations
- Resume with: `mcts-llm-alpha resume --checkpoint [checkpoint_file]`
- Located in `checkpoint_dir` (configurable)