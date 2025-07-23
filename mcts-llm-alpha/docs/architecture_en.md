# MCTS-LLM Alpha Mining Architecture

## Overview

The MCTS-LLM Alpha Mining Framework combines Monte Carlo Tree Search with Large Language Models to discover novel alpha factors for quantitative trading. This document describes the system architecture and key design decisions.

**Latest Update (2025-01-22)**: The system has completed 14 fixes to ensure full compliance with the original paper's algorithm. Major improvements include Virtual Action mechanism, unified 0-10 scoring range, symbolic parameter generation, and Few-shot learning.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   CLI/API       │────▶│   MCTS Engine   │────▶│   LLM Client    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Config Manager  │     │ Formula Processor│     │ Prompt Templates │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                         
         ▼                       ▼                         
┌─────────────────┐     ┌─────────────────┐     
│  Data Provider  │     │    Evaluator     │     
└─────────────────┘     └─────────────────┘     
```

## Component Descriptions

### 1. MCTS Engine (`mcts/`)

The core search algorithm that explores the alpha formula space:

- **MCTSNode**: Represents nodes in the search tree, tracking formulas and their scores (0-10 range)
  - UCT calculation includes normalization mechanism
  - Supports maximum 2 expansions per dimension
- **MCTSSearch**: Implements the four phases of MCTS (Selection, Expansion, Simulation, Backpropagation)
  - Virtual Action mechanism allows internal nodes to continue expanding
  - Dynamic budget adjustment based on historical maximum score
  - Dimension selection uses softmax strategy with e_max=10
- **FrequentSubtreeMiner**: Identifies overused patterns to maintain diversity
  - Only mines root genes (subtrees starting from fields)
  - Parameter values abstracted to symbol 't'

### 2. LLM Integration (`llm/`)

Interface with language models for formula generation:

- **LLMClient**: OpenAI API wrapper with retry logic
  - Symbolic parameter generation mechanism (w1, w2, t, etc.)
  - Parameter candidate evaluation and selection
- **Prompts**: Template management for different generation contexts
  - Integrated node context information
  - Few-shot example support
- **ExampleSelector**: Dimension-specific example selection strategies
  - Effectiveness/Stability: Relevance filtering
  - Diversity: Lowest correlation selection
  - Turnover/Overfitting: Zero-shot
- Two-step generation process: Portrait → Symbolic formula → Parameter instantiation

### 3. Formula Processing (`formula/`)

Handles formula manipulation and validation:

- **Cleaner**: Removes LaTeX, markdown, and other artifacts
- **Fixer**: Converts operators and adds missing parameters
- **Validator**: Syntax and operator validation

### 4. Evaluation System (`evaluation/`)

Multi-dimensional formula evaluation:

- **ComprehensiveEvaluator**: Coordinates overall evaluation process
  - Only uses Qlib for real evaluation (simulation mode removed)
- **RelativeRankingEvaluator**: Relative ranking mechanism
  - Maintains valid alpha repository (effectiveness >= 3.0)
  - Dynamic evaluation standard improvement
  - All scores unified to 0-10 range
- **Dimension Evaluators**: Specific implementations for each dimension
  - Effectiveness: Signal strength
  - Stability: Robustness
  - Turnover: Trading frequency
  - Diversity: Uniqueness
  - Overfitting: Generalization risk (default 5.0)

### 5. Data Layer (`data/`)

Abstracts data access for flexibility:

- **DataProvider**: Market data access interface
- **QlibDataProvider**: Qlib-based implementation
- **MarketDataManager**: High-level data management

### 6. Configuration Management (`config/`)

Centralized configuration management:

- YAML-based configuration files
- Pydantic models for validation
- Environment variable support
- **scoring_config.py**: Unified scoring system configuration
  - 0-10 scoring range definitions
  - UCT normalization parameters
  - Dimension thresholds (effectiveness: 3.0, diversity: 2.0, overall: 5.0)

## Key Design Patterns

### 1. Dependency Injection

The MCTS engine accepts formula generation/evaluation functions as parameters, decoupling the algorithm from implementations:

```python
mcts = MCTSSearch(
    formula_generator=generator_func,
    formula_refiner=refiner_func,
    formula_evaluator=evaluator_func
)
```

### 2. Strategy Pattern

Multiple data providers implement the same interface, allowing easy switching between Qlib and other data sources.

### 3. Template Method

The two-step formula generation (portrait → formula) follows a template pattern with customizable prompts.

### 4. Repository Pattern

The alpha repository manages discovered formulas with automatic size control and diversity maintenance.

## Data Flow

1. **Initialization**: Load config → Initialize data provider → Set up LLM client
2. **Search Loop**:
   - Select node using UCT
   - Generate refined formula via LLM
   - Evaluate formula on market data
   - Update tree statistics
3. **Output**: Save best formulas and search checkpoint

## Multi-dimensional Evaluation

Formulas are evaluated across five dimensions (0-10 scale):

1. **Effectiveness**: Predictive power and signal strength (threshold: 3.0)
2. **Stability**: Consistency across market conditions
3. **Turnover**: Trading frequency impact (lower is better)
4. **Diversity**: Uniqueness compared to existing factors (threshold: 2.0)
5. **Overfitting**: Generalization ability (default: 5.0)

Overall threshold: 5.0

## Extension Points

The architecture supports multiple extension mechanisms:

1. **Custom Operators**: Add new operators in `formula/fixer.py`
2. **Alternative LLMs**: Implement new clients in `llm/`
3. **Data Sources**: Create new providers implementing `DataProvider`
4. **Evaluation Metrics**: Extend `evaluation/metrics/`
5. **Search Strategies**: Modify MCTS parameters or selection strategies

## Performance Considerations

- **Caching**: Formula evaluations are cached to avoid redundant computation
- **Checkpointing**: Periodic saves enable long searches to resume
- **Batching**: LLM calls can be batched for efficiency
- **Memory Management**: Repository size limits prevent unbounded growth

## Security Considerations

- API keys loaded from environment variables
- No hardcoded credentials
- Formula validation prevents code injection
- Sandboxed evaluation environment

## Deployment Architecture

### Local Development
```
├── .env                 # Environment variables
├── config/             # Configuration files
├── logs/              # Log files
├── checkpoints/       # Search checkpoints
└── results/           # Alpha discoveries
```

### Production Deployment
- Containerized with Docker
- GPU support for Qlib acceleration
- Distributed evaluation possible
- Result persistence to database

## Monitoring and Observability

- Structured logging throughout
- Metrics collection for:
  - Formula generation success rate
  - Evaluation performance
  - Cache hit rates
  - Search tree statistics
- Checkpoint analysis tools

## Future Enhancements

1. **Distributed Search**: Multiple MCTS trees in parallel
2. **Online Learning**: Adapt to market regime changes
3. **Multi-market Support**: Beyond CSI300
4. **Real-time Evaluation**: Streaming data integration
5. **AutoML Integration**: Hyperparameter optimization