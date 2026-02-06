"""Centralized logging configuration"""
import logging
import sys
from utils.config import LOG_CONFIG

def setup_logger(name):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_CONFIG['level'])
    
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(LOG_CONFIG['log_file'])
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(LOG_CONFIG['format'])
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
