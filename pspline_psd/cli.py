"""Console script for pspline_psd."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("pspline_psd")
    click.echo("=" * len("pspline_psd"))
    click.echo("p-spline PSD generator")


if __name__ == "__main__":
    main()  # pragma: no cover
