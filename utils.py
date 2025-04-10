import logging


def setup_logging():
    class ColorFormatter(logging.Formatter):
        LEVEL_COLORS = {
            logging.DEBUG: "\033[94m",  # Blue
            logging.INFO: "\033[92m",  # Green
            logging.WARNING: "\033[93m",  # Yellow
            logging.ERROR: "\033[91m",  # Red
            logging.CRITICAL: "\033[95m",  # Magenta
        }
        RESET_COLOR = "\033[0m"

        def format(self, record):
            log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET_COLOR)
            record.levelname = f"{log_color}{record.levelname}{self.RESET_COLOR}"
            return super().format(record)

    formatter = ColorFormatter("%(levelname)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
