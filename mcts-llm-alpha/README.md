# MCTS-LLM Alpha挖掘框架

基于LLM的蒙特卡洛树搜索框架，用于量化金融中的公式化Alpha因子挖掘。

## 概述

本项目实现了一个先进的Alpha因子发现系统，结合了：
- 蒙特卡洛树搜索（MCTS）进行系统化探索
- 大语言模型（GPT-4）进行创造性公式生成
- 跨有效性、稳定性、换手率、多样性和过拟合的多维度评估
- 频繁子树分析（FSA）进行模式规避

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd mcts-llm-alpha

# 开发模式安装
pip install -e ".[dev]"

# 复制环境变量
cp .env.example .env
# 编辑.env添加您的OPENAI_API_KEY和QLIB_PROVIDER_URI
```

## 快速开始

### 1. 安装Qlib（必需）

系统需要Qlib来访问真实市场数据：

```bash
# 安装Qlib
pip install qlib

# 下载CSI300数据（中国A股）
python -m qlib.run.get_data qlib_data --target_dir G:/workspace/qlib_bin/qlib_bin --region cn
```

⚠️ **重要**：如果没有正确安装Qlib，系统将使用模拟数据，导致评估结果不准确。详细设置说明请参考 [QLIB_SETUP.md](QLIB_SETUP.md)。

### 2. 配置环境变量

编辑`.env`文件：

```bash
OPENAI_API_KEY=your_api_key_here
QLIB_PROVIDER_URI=G:/workspace/qlib_bin/qlib_bin
```

### 3. 运行系统

```bash
# 直接运行脚本（推荐用于调试）
python run_search.py --iterations 50

# 或使用CLI工具
mcts-llm-alpha search --iterations 50
```

## 使用方法

### 基础搜索

```bash
# 使用默认参数运行MCTS搜索
mcts-llm-alpha search

# 使用自定义配置运行
mcts-llm-alpha search --config config/experiment.yaml
```

### 从检查点恢复

```bash
# 从特定检查点恢复
mcts-llm-alpha resume --checkpoint mcts_checkpoint_20240101_120000.pkl
```

### 评估Alpha仓库

```bash
# 评估保存的alpha因子
mcts-llm-alpha evaluate --input results/alpha_repository.json
```

## 项目结构

```
mcts-llm-alpha/
├── src/mcts_llm_alpha/
│   ├── config/          # 配置管理
│   ├── llm/            # LLM集成（提示词、客户端）
│   ├── mcts/           # MCTS算法（节点、搜索、FSA）
│   ├── formula/        # 公式处理（清理器、修复器、评估器）
│   ├── data/           # 数据提供者（Qlib封装）
│   └── cli.py          # 命令行接口
├── notebooks/          # 实验用Jupyter笔记本
├── tests/             # 单元测试
└── docs/              # 文档
```

## 核心特性

1. **两步生成过程**：
   - 首先生成带有高层描述的"Alpha画像"
   - 然后将画像转换为具体的数学公式

2. **多维度优化**：
   - 有效性：信号强度和预测准确性
   - 稳定性：跨市场条件的稳健性
   - 换手率：交易频率优化
   - 多样性：与现有因子的独特性
   - 过拟合：泛化能力

3. **智能公式修复**：
   - 自动操作符映射和参数修复
   - 全面的验证和语法检查

4. **模式规避**：
   - 频繁子树分析防止过度使用常见模式
   - 保持alpha仓库的多样性

## 配置

参见`config/default.yaml`了解所有可用参数：

- `max_iterations`：最大MCTS迭代次数
- `exploration_constant`：UCT探索参数
- `dimension_temperature`：维度选择的Softmax温度
- `max_depth`：最大树深度
- `checkpoint_freq`：检查点保存频率

## 开发

```bash
# 运行测试
pytest

# 格式化代码
black src tests
isort src tests

# 代码检查
ruff check src tests

# 类型检查
mypy src
```

## 许可证

[您的许可证]