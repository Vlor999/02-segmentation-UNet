from model import config
from argparse import ArgumentParser
from loguru import logger


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--num-epochs",
        help="Please choose the number of epochs.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Please choose the size of the batches.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--device",
        help="Please select a device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        type=str,
    )
    parser.add_argument(
        "--num-workers",
        help="Please select the number of workers. Won't work if device is mps unless you use --force",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--force",
        help="Force the configuration",
        default=False,
        type=bool,
        choices=[True, False],
    )
    parser.add_argument(
        "--model-name",
        help="Give the model name",
        default="UNet",
        choices=["UNet", "Simple", "HourGlass"],
        type=str,
    )

    args = parser.parse_args()

    if args.num_epochs is not None:
        config.NUM_EPOCHS = int(args.num_epochs)
        logger.info(f"Number of epochs set to: {config.NUM_EPOCHS}")
    if args.batch_size is not None:
        config.BATCH_SIZE = int(args.batch_size)
        logger.info(f"Batch size set to: {config.BATCH_SIZE}")
    config.DEVICE = args.device
    logger.info(f"Device set to: {config.DEVICE}")
    if args.device == "mps" and not args.force:
        config.NUM_WORKERS = 0
        logger.info("Using MPS device without force, setting num_workers to 0")
    else:
        config.NUM_WORKERS = args.num_workers
        logger.info(f"Number of workers set to: {config.NUM_WORKERS}")
    config.MODEL_NAME = args.model_name

    return args
