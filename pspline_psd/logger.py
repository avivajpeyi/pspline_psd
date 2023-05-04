import sys
import warnings

from loguru import logger

NAME = "<blue>P-Spline PSD</blue>"
TIME = "{time:DD/MM HH:mm:ss}"

logger.remove(0)
logger.add(
    sys.stderr,
    format=(
        f"|{NAME}|{TIME}|"
        "{level}| <green>{message}</green>"
    ),
    colorize=True,
    level="INFO",
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
