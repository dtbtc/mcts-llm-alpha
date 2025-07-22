#!/usr/bin/env python3
"""
MCTS-LLM Alpha挖掘系统的命令行界面。

该模块提供主入口点和CLI命令，用于：
- 运行MCTS搜索alpha公式
- 从检查点恢复搜索
- 评估alpha仓库
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Config, MCTSConfig, LLMConfig
from .mcts import MCTSSearch, MCTSNode
from .llm import LLMInterface
from .formula import FormulaGenerator, FormulaEvaluator
from .data import DataLoader


# 设置rich控制台以获得美观的输出
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """设置日志配置。"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def load_config(config_path: Optional[Path] = None) -> Config:
    """从文件加载配置或使用默认值。"""
    if config_path and config_path.exists():
        console.print(f"[cyan]Loading config from {config_path}[/cyan]")
        return Config.from_yaml(str(config_path))
    else:
        console.print("[yellow]Using default configuration[/yellow]")
        return Config()


def search_command(args: argparse.Namespace) -> None:
    """运行MCTS搜索alpha公式。"""
    console.print("[bold green]Starting MCTS Search for Alpha Formulas[/bold green]")
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化组件
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # 加载数据
        task = progress.add_task("Loading market data...", total=None)
        data_loader = DataLoader(config.data_config)
        market_data = data_loader.load_data(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols.split(',') if args.symbols else None
        )
        progress.update(task, completed=True)
        
        # 初始化LLM
        task = progress.add_task("Initializing LLM...", total=None)
        llm = LLMInterface(config.llm_config)
        progress.update(task, completed=True)
        
        # 初始化公式组件
        task = progress.add_task("Setting up formula evaluation...", total=None)
        formula_generator = FormulaGenerator(llm)
        formula_evaluator = FormulaEvaluator(market_data)
        progress.update(task, completed=True)
        
        # 初始化MCTS
        task = progress.add_task("Initializing MCTS...", total=None)
        mcts = MCTSSearch(
            config=config.mcts_config,
            llm=llm,
            formula_generator=formula_generator,
            formula_evaluator=formula_evaluator
        )
        progress.update(task, completed=True)
    
    # 运行搜索
    console.print(f"\n[cyan]Running search for {args.iterations} iterations[/cyan]")
    console.print(f"[cyan]Output directory: {args.output}[/cyan]\n")
    
    try:
        best_node = mcts.search(
            iterations=args.iterations,
            checkpoint_dir=Path(args.output) / "checkpoints" if args.checkpoint else None
        )
        
        # 保存结果
        save_results(best_node, Path(args.output), mcts.get_top_formulas(args.top_k))
        
        console.print("\n[bold green]Search completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted by user[/yellow]")
        if args.checkpoint:
            console.print(f"[yellow]Checkpoint saved. Use 'resume' command to continue.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error during search: {e}[/bold red]")
        raise


def resume_command(args: argparse.Namespace) -> None:
    """从检查点恢复MCTS搜索。"""
    console.print("[bold green]Resuming MCTS Search from Checkpoint[/bold green]")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[bold red]Checkpoint not found: {checkpoint_path}[/bold red]")
        sys.exit(1)
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化组件 (similar to search_command)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # 加载数据
        task = progress.add_task("Loading market data...", total=None)
        data_loader = DataLoader(config.data_config)
        
        # 如果可用，从检查点提取日期范围
        checkpoint_data = json.loads(checkpoint_path.read_text())
        start_date = checkpoint_data.get('start_date', args.start_date)
        end_date = checkpoint_data.get('end_date', args.end_date)
        
        market_data = data_loader.load_data(
            start_date=start_date,
            end_date=end_date
        )
        progress.update(task, completed=True)
        
        # 初始化组件
        task = progress.add_task("Initializing components...", total=None)
        llm = LLMInterface(config.llm_config)
        formula_generator = FormulaGenerator(llm)
        formula_evaluator = FormulaEvaluator(market_data)
        
        mcts = MCTSSearch(
            config=config.mcts_config,
            llm=llm,
            formula_generator=formula_generator,
            formula_evaluator=formula_evaluator
        )
        progress.update(task, completed=True)
        
        # 加载检查点
        task = progress.add_task("Loading checkpoint...", total=None)
        mcts.load_checkpoint(checkpoint_path)
        progress.update(task, completed=True)
    
    # 继续搜索
    remaining_iterations = args.iterations
    console.print(f"\n[cyan]Continuing search for {remaining_iterations} more iterations[/cyan]\n")
    
    try:
        best_node = mcts.search(
            iterations=remaining_iterations,
            checkpoint_dir=checkpoint_path.parent
        )
        
        # 保存结果
        save_results(best_node, Path(args.output), mcts.get_top_formulas(args.top_k))
        
        console.print("\n[bold green]Search resumed and completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error during resumed search: {e}[/bold red]")
        raise


def evaluate_command(args: argparse.Namespace) -> None:
    """评估alpha仓库。"""
    console.print("[bold green]Evaluating Alpha Repository[/bold green]")
    
    repo_path = Path(args.repository)
    if not repo_path.exists():
        console.print(f"[bold red]Repository not found: {repo_path}[/bold red]")
        sys.exit(1)
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载市场数据
    console.print("[cyan]Loading market data...[/cyan]")
    data_loader = DataLoader(config.data_config)
    market_data = data_loader.load_data(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 从仓库加载公式
    console.print("[cyan]Loading formulas from repository...[/cyan]")
    formulas = load_alpha_repository(repo_path)
    
    if not formulas:
        console.print("[yellow]No formulas found in repository[/yellow]")
        return
    
    # 评估公式
    console.print(f"[cyan]Evaluating {len(formulas)} formulas...[/cyan]")
    formula_evaluator = FormulaEvaluator(market_data)
    
    results = []
    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating formulas...", total=len(formulas))
        
        for name, formula in formulas.items():
            try:
                metrics = formula_evaluator.evaluate(formula)
                results.append({
                    'name': name,
                    'formula': formula,
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'annual_return': metrics.get('annual_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'ic': metrics.get('ic', 0),
                    'valid': True
                })
            except Exception as e:
                results.append({
                    'name': name,
                    'formula': formula,
                    'error': str(e),
                    'valid': False
                })
            
            progress.update(task, advance=1)
    
    # 显示结果
    display_evaluation_results(results)
    
    # 保存详细结果
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path / "evaluation_results.csv", index=False)
        
        with open(output_path / "evaluation_summary.json", 'w') as f:
            summary = {
                'total_formulas': len(formulas),
                'valid_formulas': sum(1 for r in results if r.get('valid', False)),
                'average_sharpe': df[df['valid']]['sharpe_ratio'].mean() if any(df['valid']) else 0,
                'best_formula': df[df['valid']].nlargest(1, 'sharpe_ratio').to_dict('records')[0] if any(df['valid']) else None
            }
            json.dump(summary, f, indent=2)
        
        console.print(f"\n[green]Results saved to {output_path}[/green]")


def save_results(best_node: MCTSNode, output_dir: Path, top_formulas: list) -> None:
    """将搜索结果保存到输出目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存最佳公式
    with open(output_dir / "best_formula.json", 'w') as f:
        json.dump({
            'formula': best_node.formula,
            'value': best_node.value,
            'visits': best_node.visits,
            'metrics': best_node.evaluation_metrics
        }, f, indent=2)
    
    # 保存顶级公式
    with open(output_dir / "top_formulas.json", 'w') as f:
        json.dump(top_formulas, f, indent=2)
    
    # 以alpha仓库格式保存
    alpha_repo = {}
    for i, formula_info in enumerate(top_formulas):
        alpha_repo[f"alpha_{i+1:03d}"] = formula_info['formula']
    
    with open(output_dir / "alpha_repository.json", 'w') as f:
        json.dump(alpha_repo, f, indent=2)
    
    console.print(f"[green]Results saved to {output_dir}[/green]")


def load_alpha_repository(repo_path: Path) -> Dict[str, str]:
    """从仓库文件加载alpha公式。"""
    if repo_path.suffix == '.json':
        with open(repo_path) as f:
            return json.load(f)
    elif repo_path.suffix == '.csv':
        df = pd.read_csv(repo_path)
        if 'name' in df.columns and 'formula' in df.columns:
            return dict(zip(df['name'], df['formula']))
        else:
            raise ValueError("CSV must have 'name' and 'formula' columns")
    else:
        raise ValueError(f"Unsupported repository format: {repo_path.suffix}")


def display_evaluation_results(results: list) -> None:
    """在美观的表格中显示评估结果。"""
    table = Table(title="Alpha Evaluation Results")
    
    table.add_column("Name", style="cyan")
    table.add_column("Formula", style="white", max_width=40)
    table.add_column("Sharpe", style="green")
    table.add_column("Return", style="blue")
    table.add_column("Drawdown", style="red")
    table.add_column("IC", style="yellow")
    table.add_column("Status", style="bold")
    
    for result in sorted(results, key=lambda x: x.get('sharpe_ratio', -999), reverse=True):
        if result.get('valid', False):
            table.add_row(
                result['name'],
                result['formula'][:40] + "..." if len(result['formula']) > 40 else result['formula'],
                f"{result['sharpe_ratio']:.3f}",
                f"{result['annual_return']:.2%}",
                f"{result['max_drawdown']:.2%}",
                f"{result['ic']:.3f}",
                "[green]Valid[/green]"
            )
        else:
            table.add_row(
                result['name'],
                result['formula'][:40] + "..." if len(result['formula']) > 40 else result['formula'],
                "-",
                "-",
                "-",
                "-",
                f"[red]Error: {result.get('error', 'Unknown')}[/red]"
            )
    
    console.print(table)


def main():
    """CLI的主入口点。"""
    parser = argparse.ArgumentParser(
        description="MCTS-LLM Alpha挖掘系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 运行新搜索
  mcts-llm-alpha search --iterations 1000 --output results/

  # 从检查点恢复
  mcts-llm-alpha resume --checkpoint results/checkpoints/latest.json --iterations 500

  # 评估alpha仓库
  mcts-llm-alpha evaluate --repository alphas.json --start-date 2023-01-01 --end-date 2023-12-31
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-c', '--config', type=Path, help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='Run MCTS search for alpha formulas')
    search_parser.add_argument('-i', '--iterations', type=int, default=100, 
                              help='Number of MCTS iterations (default: 100)')
    search_parser.add_argument('-o', '--output', type=str, default='results/',
                              help='Output directory (default: results/)')
    search_parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    search_parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    search_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    search_parser.add_argument('--checkpoint', action='store_true', 
                              help='Enable checkpointing during search')
    search_parser.add_argument('--top-k', type=int, default=10,
                              help='Number of top formulas to save (default: 10)')
    
    # 恢复命令
    resume_parser = subparsers.add_parser('resume', help='Resume search from checkpoint')
    resume_parser.add_argument('-c', '--checkpoint', type=str, required=True,
                              help='Path to checkpoint file')
    resume_parser.add_argument('-i', '--iterations', type=int, default=100,
                              help='Additional iterations to run (default: 100)')
    resume_parser.add_argument('-o', '--output', type=str, default='results/',
                              help='Output directory (default: results/)')
    resume_parser.add_argument('--top-k', type=int, default=10,
                              help='Number of top formulas to save (default: 10)')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate alpha repository')
    eval_parser.add_argument('-r', '--repository', type=str, required=True,
                            help='Path to alpha repository (JSON or CSV)')
    eval_parser.add_argument('--start-date', type=str, required=True,
                            help='Start date for evaluation (YYYY-MM-DD)')
    eval_parser.add_argument('--end-date', type=str, required=True,
                            help='End date for evaluation (YYYY-MM-DD)')
    eval_parser.add_argument('-o', '--output', type=str,
                            help='Output directory for results')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 执行命令
    if args.command == 'search':
        search_command(args)
    elif args.command == 'resume':
        resume_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()