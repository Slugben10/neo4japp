
# Application initialization hook
import os
import sys
import json
import logging
import platform

def setup_logging():
    # Configure logging
    log_format = "[INFO] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    return logger

logger = setup_logging()
logger.info(f"Initializing application runtime hook in {__file__}")

# Find the correct app path based on platform and frozen state
if getattr(sys, 'frozen', False):
    # Running in a bundle
    if sys.platform == 'darwin':  # macOS
        # macOS app bundles have a different structure
        # Check various possible locations for the executable
        logger.info(f"Running as macOS app bundle")
        bundle_base = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
        logger.info(f"Bundle base path: {bundle_base}")
        
        app_paths = [
            os.path.dirname(sys.executable),  # MacOS/
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Resources'),  # Resources/
            bundle_base,  # Base .app directory
        ]
        
        # Find first path containing necessary directories or files
        APP_PATH = None
        for path in app_paths:
            logger.info(f"Checking path: {path}")
            if os.path.exists(path) and (
                os.path.exists(os.path.join(path, 'Neo4jDB')) or 
                os.path.exists(os.path.join(path, 'config.json')) or
                os.path.exists(os.path.join(path, 'Documents'))
            ):
                APP_PATH = path
                logger.info(f"Using path: {APP_PATH}")
                break
                
        if APP_PATH is None:
            # Fallback to default locations
            APP_PATH = bundle_base
            logger.info(f"Using fallback path: {APP_PATH}")
            
        # Create core directories if they don't exist 
        os.makedirs(os.path.join(APP_PATH, "Documents"), exist_ok=True)
        os.makedirs(os.path.join(APP_PATH, "Neo4jDB"), exist_ok=True)
        os.makedirs(os.path.join(APP_PATH, "Prompts"), exist_ok=True)
        os.makedirs(os.path.join(APP_PATH, "jre"), exist_ok=True)
        
        # Set environment variables to help find resources
        os.environ['RA_APP_PATH'] = APP_PATH
        os.environ['RA_BUNDLE_PATH'] = bundle_base
        os.environ['RA_RESOURCES_PATH'] = os.path.join(bundle_base, 'Resources')
        os.environ['RA_MACOS_PATH'] = os.path.join(bundle_base, 'MacOS')
                
    else:  # Windows/Linux
        APP_PATH = os.path.dirname(sys.executable)
        logger.info(f"Running as frozen application: {APP_PATH}")
        os.environ['RA_APP_PATH'] = APP_PATH
        
    # Load configuration to check Neo4j preservation settings
    config_path = os.path.join(APP_PATH, 'config.json')
    if os.path.exists(config_path):
        try:
            logger.info(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set environment variables for Neo4j preservation
            os.environ['PRESERVE_NEO4J_DATA'] = 'True' if config.get('preserve_neo4j_data', True) else 'False'
            os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True' if config.get('download_neo4j_if_missing', True) else 'False'
            logger.info(f"Neo4j preservation: {os.environ['PRESERVE_NEO4J_DATA']}")
        except Exception as e:
            logger.info(f"Error loading config: {e}")
            # Default to preserving data if config can't be loaded
            os.environ['PRESERVE_NEO4J_DATA'] = 'True'
            os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True'
    else:
        logger.info(f"Config file not found at: {config_path}")
        # Default to preserving data if config doesn't exist
        os.environ['PRESERVE_NEO4J_DATA'] = 'True'
        os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True'
