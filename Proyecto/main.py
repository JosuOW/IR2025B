"""
main.py
Script de alto nivel que ilustra uso de los módulos.
"""
from cli import main as cli_main
# main delega a cli.py which provides the command line interface
if __name__ == "__main__":
    cli_main()
