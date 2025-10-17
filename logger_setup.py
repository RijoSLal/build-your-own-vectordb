import logging
from logging.handlers import RotatingFileHandler


def setup_config():
    """
    this functions handle vector_db logs 
    
    """

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_file_name = "vector_db_ops.log"
    file_handler = RotatingFileHandler(log_file_name, maxBytes=1 * 1024 * 1024, backupCount=1)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))


    logging.basicConfig(
        format = log_format, 
        level = logging.INFO,
        handlers = [file_handler, console_handler]        
    )
   
