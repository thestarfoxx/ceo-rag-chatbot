import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

def load_config() -> Dict[str, Any]:
    """Load configuration from various sources."""
    
    config = {
        # OpenAI Configuration
        'openai': {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'embedding_model': os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
            'chat_model': os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o'),
            'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '1000')),
            'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        },
        
        # Database Configuration
        'database': {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'ceo_rag_db'),
            'username': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        },
        
        # Vector Search Configuration
        'vector_search': {
            'similarity_threshold': float(os.getenv('VECTOR_SIMILARITY_THRESHOLD', '0.25')),
            'max_results': int(os.getenv('VECTOR_MAX_RESULTS', '10')),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '500')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '50'))
        },
        
        # SQL Search Configuration  
        'sql_search': {
            'max_results': int(os.getenv('SQL_MAX_RESULTS', '50')),
            'timeout': int(os.getenv('SQL_TIMEOUT', '30'))
        },
        
        # Application Configuration
        'app': {
            'title': os.getenv('APP_TITLE', 'CEO RAG Chatbot'),
            'description': os.getenv('APP_DESCRIPTION', 'Intelligent Assistant for Executive Decision Making'),
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'session_timeout': int(os.getenv('SESSION_TIMEOUT', '3600'))
        },
        
        # Email Configuration
        'email': {
            'default_tone': os.getenv('EMAIL_DEFAULT_TONE', 'professional'),
            'max_draft_length': int(os.getenv('EMAIL_MAX_DRAFT_LENGTH', '2000')),
            'auto_analysis': os.getenv('EMAIL_AUTO_ANALYSIS', 'True').lower() == 'true'
        },
        
        # File Upload Configuration
        'upload': {
            'max_file_size': int(os.getenv('UPLOAD_MAX_FILE_SIZE', '100')),  # MB
            'allowed_pdf_extensions': ['.pdf'],
            'allowed_excel_extensions': ['.xlsx', '.xls', '.xlsm'],
            'upload_directory': os.getenv('UPLOAD_DIRECTORY', 'data/')
        }
    }
    
    # Try to load from config file if it exists
    config_file = Path(__file__).parent.parent.parent / 'config.yaml'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge file config with environment config
                config.update(file_config)
        except Exception as e:
            st.warning(f"Could not load config file: {e}")
    
    return config

def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate configuration and return validation results."""
    
    errors = []
    
    # Check required OpenAI configuration
    if not config['openai']['api_key']:
        errors.append("OpenAI API key is required")
    
    # Check database configuration
    db_config = config['database']
    if not all([db_config['host'], db_config['database'], db_config['username']]):
        errors.append("Database configuration is incomplete")
    
    # Check vector search configuration
    vector_config = config['vector_search']
    if vector_config['similarity_threshold'] > 1.0 or vector_config['similarity_threshold'] < 0.0:
        errors.append("Vector similarity threshold must be between 0.0 and 1.0")
    
    if vector_config['max_results'] <= 0:
        errors.append("Vector search max results must be positive")
    
    # Check upload configuration
    upload_config = config['upload']
    if upload_config['max_file_size'] <= 0:
        errors.append("Upload max file size must be positive")
    
    return len(errors) == 0, errors

def get_db_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract database configuration for the retriever."""
    db = config['database']
    return {
        'host': str(db['host']),
        'port': str(db['port']),
        'database': db['database'],
        'username': db['username'],
        'password': db['password']
    }

def create_directories(config: Dict[str, Any]):
    """Create necessary directories based on configuration."""
    upload_dir = Path(config['upload']['upload_directory'])
    
    # Create main upload directory
    upload_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (upload_dir / 'documents' / 'pdfs').mkdir(parents=True, exist_ok=True)
    (upload_dir / 'documents' / 'excel').mkdir(parents=True, exist_ok=True)
    (upload_dir / 'documents' / 'processed').mkdir(parents=True, exist_ok=True)
    (upload_dir / 'logs').mkdir(exist_ok=True)

def save_config(config: Dict[str, Any], config_path: Optional[Path] = None):
    """Save configuration to file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Could not save configuration: {e}")
        return False

def get_upload_config() -> Dict[str, Any]:
    """Get upload-specific configuration."""
    config = load_config()
    return config['upload']

def get_email_config() -> Dict[str, Any]:
    """Get email-specific configuration."""
    config = load_config()
    return config['email']

def get_vector_config() -> Dict[str, Any]:
    """Get vector search configuration."""
    config = load_config()
    return config['vector_search']

def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    config = load_config()
    return config['app']['debug']

# Configuration validation for Streamlit
def display_config_status():
    """Display configuration status in Streamlit."""
    config = load_config()
    is_valid, errors = validate_config(config)
    
    if is_valid:
        st.success("✅ Configuration is valid")
    else:
        st.error("❌ Configuration errors found:")
        for error in errors:
            st.error(f"• {error}")
    
    return is_valid, config

def display_config_details(config: Dict[str, Any]):
    """Display configuration details in an expandable section."""
    with st.expander("🔧 Configuration Details"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("OpenAI")
            st.write(f"**Model:** {config['openai']['chat_model']}")
            st.write(f"**Embeddings:** {config['openai']['embedding_model']}")
            st.write(f"**Max Tokens:** {config['openai']['max_tokens']}")
            st.write(f"**Temperature:** {config['openai']['temperature']}")
            st.write(f"**API Key:** {'✅ Set' if config['openai']['api_key'] else '❌ Missing'}")
            
            st.subheader("Vector Search")
            st.write(f"**Similarity Threshold:** {config['vector_search']['similarity_threshold']}")
            st.write(f"**Max Results:** {config['vector_search']['max_results']}")
            st.write(f"**Chunk Size:** {config['vector_search']['chunk_size']}")
            st.write(f"**Chunk Overlap:** {config['vector_search']['chunk_overlap']}")
        
        with col2:
            st.subheader("Database")
            st.write(f"**Host:** {config['database']['host']}")
            st.write(f"**Port:** {config['database']['port']}")
            st.write(f"**Database:** {config['database']['database']}")
            st.write(f"**Username:** {config['database']['username']}")
            st.write(f"**Password:** {'✅ Set' if config['database']['password'] else '❌ Missing'}")
            
            st.subheader("Application")
            st.write(f"**Debug Mode:** {config['app']['debug']}")
            st.write(f"**Log Level:** {config['app']['log_level']}")
            st.write(f"**Session Timeout:** {config['app']['session_timeout']}s")
            st.write(f"**Max Upload Size:** {config['upload']['max_file_size']}MB")

# Environment variable helpers
def set_env_var(key: str, value: str):
    """Set environment variable."""
    os.environ[key] = value

def get_env_var(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with default."""
    return os.getenv(key, default)

def create_env_file_template():
    """Create a template .env file."""
    template = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ceo_rag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Vector Search Configuration
VECTOR_SIMILARITY_THRESHOLD=0.25
VECTOR_MAX_RESULTS=10
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# SQL Search Configuration
SQL_MAX_RESULTS=50
SQL_TIMEOUT=30

# Application Configuration
APP_TITLE=CEO RAG Chatbot
APP_DESCRIPTION=Intelligent Assistant for Executive Decision Making
DEBUG=False
LOG_LEVEL=INFO
SESSION_TIMEOUT=3600

# Email Configuration
EMAIL_DEFAULT_TONE=professional
EMAIL_MAX_DRAFT_LENGTH=2000
EMAIL_AUTO_ANALYSIS=True

# Upload Configuration
UPLOAD_MAX_FILE_SIZE=100
UPLOAD_DIRECTORY=data/
"""
    
    env_file = Path(__file__).parent.parent.parent / '.env.example'
    with open(env_file, 'w') as f:
        f.write(template)
    
    return env_file
