import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, workspace_dir: str, level: str = "INFO") -> logging.Logger:
    """
    Standardizes log formats across the memory system, enabling rotating
    file handlers and streaming.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicated handlers if called multiple times
    if logger.hasHandlers():
        return logger
        
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    log_dir = os.path.join(workspace_dir, "memory", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "system.log")
    
    # Max size 5MB, keep up to 3 backups
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(numeric_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
