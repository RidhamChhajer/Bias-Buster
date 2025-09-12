import os
from dotenv import load_dotenv
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent.parent
load_dotenv(ROOT_DIR / '.env')

class Config:
    """Configuration class for Bias Buster application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-nano-2025-08-07')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', 1500))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', 0.3))
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'bias-buster-secret-2025')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'apps/web-app/uploads')
    REPORTS_FOLDER = os.getenv('REPORTS_FOLDER', 'apps/web-app/reports')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
    
    @staticmethod
    def validate_config():
        """Validate required configuration"""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please add it to your .env file")
        return True

    @classmethod
    def init_app(cls, app):
        """Initialize Flask app with configuration"""
        app.config.from_object(cls)
