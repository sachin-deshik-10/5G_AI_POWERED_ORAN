import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

# Default configuration
LOG_DIR = "logs"
LOG_LEVEL = logging.INFO

class Logger:
    def __init__(self, log_file_name, logger_name=None):
        if logger_name is None:
            logger_name = __name__
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(LOG_LEVEL)
        self.log_file_name = log_file_name
        
        # Create logs directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file_path = Path(LOG_DIR) / self.log_file_name
        self._configure_logger()

    def _configure_logger(self):
        # Clear existing handlers to avoid duplication
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOG_LEVEL)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # File handler
        file_handler = TimedRotatingFileHandler(
            filename=self.log_file_path,
            when="midnight",
            backupCount=7,
            utc=True,
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, message, level="info"):
        """Generic log method"""
        getattr(self.logger, level.lower())(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

def get_logger(name):
    """Factory function to get a logger instance"""
    return logging.getLogger(name)


