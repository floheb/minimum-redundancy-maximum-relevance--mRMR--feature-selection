
import logging

# ─────────────────────────────── LOGGER SETUP ───────────────────────────────

class ColoredLogger:
    """Simple colored logger for INFO/WARNING/ERROR levels."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Color codes
        self.colors = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m', 
            'RED': '\033[91m',
            'RESET': '\033[0m'
        }
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log info message in green."""
        colored_msg = f"{self.colors['GREEN']}[INFO]{self.colors['RESET']} {message}"
        self.logger.info(colored_msg)
    
    def warning(self, message: str) -> None:
        """Log warning message in yellow."""
        colored_msg = f"{self.colors['YELLOW']}[WARNING]{self.colors['RESET']} {message}"
        self.logger.warning(colored_msg)
    
    def error(self, message: str) -> None:
        """Log error message in red."""
        colored_msg = f"{self.colors['RED']}[ERROR]{self.colors['RESET']} {message}"
        self.logger.error(colored_msg)
