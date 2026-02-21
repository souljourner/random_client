"""CLI entry point for TruFor client — sends images to the detection server."""

import logging
import os
import sys

import click
import yaml

from pipeline import run_pipeline


@click.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True), help="Path to input CSV (ticketId, image_path)")
@click.option("--output", "output_csv", default=None, type=click.Path(), help="Path to output CSV [default: ../output/results.csv]")
@click.option("--config", "config_path", default="config.yaml", type=click.Path(exists=True), help="Path to client config.yaml")
@click.option("--limit", default=None, type=int, help="Process only first N rows")
@click.option("--no-resume", is_flag=True, help="Ignore checkpoint, start fresh")
@click.option("--dry-run", is_flag=True, help="Validate CSV and test image files without sending to server")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(input_csv, output_csv, config_path, limit, no_resume, dry_run, verbose):
    """TruFor Client — Send images to the detection server.

    Reads a CSV of local image paths, sends each to the TruFor server,
    and collects forgery scores, explanations, and domain tags into an output CSV.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Default output path
    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(__file__), "..", "output", "results.csv")

    logging.info("Input:  %s", input_csv)
    logging.info("Output: %s", output_csv)
    logging.info("Config: %s", config_path)
    logging.info("Server: %s", config["server"]["url"])

    try:
        run_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            config=config,
            limit=limit,
            no_resume=no_resume,
            dry_run=dry_run,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted. Progress has been checkpointed.")
        sys.exit(1)
    except Exception as e:
        logging.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
