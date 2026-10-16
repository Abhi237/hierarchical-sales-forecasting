from __future__ import annotations

import click

from .config import load_config


@click.group()
@click.option("--config", default=None, help="path to config.yaml")
@click.pass_context
def main(ctx, config):
    """hierarchical-sales-forecasting pipeline."""
    ctx.obj = load_config(config)


@main.command()
@click.pass_obj
def sql(cfg):
    """Run the numbered DuckDB SQL scripts and write the R inputs."""
    from .pipeline import run_sql
    click.echo(run_sql(cfg))


@main.command()
@click.option("--layer", "layers", multiple=True, type=click.Choice(["hier", "item"]))
@click.pass_obj
def backtest(cfg, layers):
    """Rolling-origin backtest of the Python models."""
    from .pipeline import run_backtest
    run_backtest(cfg, list(layers) or None)


@main.command("r")
@click.pass_obj
def run_r_cmd(cfg):
    """fable ETS/ARIMA + reconciliation + residual tests (Rscript)."""
    from .pipeline import run_r
    run_r(cfg)


@main.command()
@click.pass_obj
def evaluate(cfg):
    """Score every forecast with one metric code path; write artifacts/."""
    from .pipeline import run_evaluate
    run_evaluate(cfg)


@main.command()
@click.pass_obj
def export(cfg):
    """Write the Tableau CSVs to tableau/data/."""
    from .pipeline import run_exports
    for r in run_exports(cfg):
        click.echo(f"{r['file']:<36} {r['rows']:>9} rows {r['bytes'] / 1e6:8.2f} MB")


@main.command()
@click.pass_obj
def figures(cfg):
    """Static matplotlib figures to docs/figures/."""
    from .plots import make_figures
    make_figures(cfg)


if __name__ == "__main__":
    main()
