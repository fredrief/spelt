import click
from pathlib import Path
from spelt.inspector import create_app

@click.command()
@click.option('--directory', '-d', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Root directory for signal browsing')
def inspect(directory=None):
    """Launch the Spelt signal inspector."""
    # Use current working directory if no directory specified
    root_dir = Path(directory) if directory else Path.cwd()

    # Create and configure the Flask app
    app = create_app()
    app.config['DEFAULT_ROOT_DIR'] = str(root_dir)

    # Run the app
    app.run(debug=True)

if __name__ == '__main__':
    inspect()
