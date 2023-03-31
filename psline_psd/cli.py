"""Console script for psline_psd."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("psline_psd")
    click.echo("=" * len("psline_psd"))
    click.echo("p-spline PSD generator")


if __name__ == "__main__":
    main()  # pragma: no cover
