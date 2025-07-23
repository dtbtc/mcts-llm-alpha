# Installation Guide

This guide covers the complete installation process for the MCTS-LLM Alpha Mining Framework.

## Prerequisites

### System Requirements
- **OS**: Linux, Windows, or macOS
- **Python**: 3.8 or higher
- **Memory**: 16GB RAM minimum (32GB recommended)
- **Storage**: 10GB free space (for Qlib data)
- **GPU**: Optional but recommended for faster Qlib computations

### Required Accounts
- **OpenAI Account** with GPT-4 API access
- **GitHub Account** (for cloning private repos)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dtbtc/mcts-llm-alpha.git
cd mcts-llm-alpha
```

### 2. Set Up Python Environment

We recommend using Conda for environment management:

```bash
# Create new environment
conda create -n alphagen_dev python=3.8
conda activate alphagen_dev

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Package

```bash
cd mcts-llm-alpha
pip install -e ".[dev]"
```

This installs the package in development mode with all dependencies.

### 4. Set Up Qlib

Qlib is required for market data and backtesting:

```bash
# Install Qlib
pip install qlib

# Download CSI300 data (Chinese A-share market)
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data/cn_data --region cn

# For US market data (optional)
python -m qlib.run.get_data qlib_data --target_dir ./qlib_data/us_data --region us
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here

# Qlib Data Path
QLIB_PROVIDER_URI=/absolute/path/to/qlib_data/cn_data

# Optional: Logging
LOG_LEVEL=INFO

# Optional: Results Directory
RESULTS_DIR=./results
CHECKPOINT_DIR=./checkpoints
```

### 6. Verify Installation

Run the test suite to ensure everything is set up correctly:

```bash
# Run basic tests
pytest tests/test_config.py -v

# Test Qlib connection
python -c "import qlib; qlib.init(provider_uri='./qlib_data/cn_data'); print('Qlib OK')"

# Test OpenAI connection
python -c "from openai import OpenAI; client = OpenAI(); print('OpenAI OK')"
```

## Common Installation Issues

### Issue: Qlib Installation Fails

**Solution**: Install build tools first
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential

# On Windows
# Install Visual Studio Build Tools

# On macOS
xcode-select --install
```

### Issue: OpenAI API Key Not Found

**Solution**: Ensure the key is properly set
```bash
# Linux/macOS
export OPENAI_API_KEY=sk-your-key

# Windows
set OPENAI_API_KEY=sk-your-key
```

### Issue: Memory Errors During Evaluation

**Solution**: Reduce batch size in configuration
```yaml
# config/default.yaml
evaluation:
  batch_size: 100  # Reduce from default 1000
```

## Docker Installation (Alternative)

For a containerized setup:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

CMD ["python", "mcts-llm-alpha/run_search.py"]
```

Build and run:
```bash
docker build -t mcts-llm-alpha .
docker run -it --env-file .env mcts-llm-alpha
```

## Next Steps

After successful installation:
1. Read the [Quick Start Tutorial](Quick-Start-Tutorial)
2. Review [Configuration Guide](Configuration-Guide)
3. Run your first alpha mining search

For any issues, please check [Common Issues](Common-Issues) or open a GitHub issue.