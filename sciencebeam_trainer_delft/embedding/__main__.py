import logging

from sciencebeam_trainer_delft.embedding.cli import main


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(level='INFO')

    main()
