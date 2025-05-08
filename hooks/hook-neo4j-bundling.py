
# Neo4j bundling hook
import os
import sys
import shutil
import logging
import platform

def setup_logging():
    # Configure logging
    log_format = "[INFO] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    return logger

logger = setup_logging()
logger.info("Initializing Neo4j bundling hook")

# Find the correct app path based on platform and frozen state
if getattr(sys, 'frozen', False):
    if sys.platform == 'darwin':  # macOS
        # macOS app bundles have a different structure
        # Use the environment variable from the app hook if available
        APP_PATH = os.environ.get('RA_APP_PATH')
        if not APP_PATH:
            bundle_base = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
            APP_PATH = bundle_base
            logger.info(f"Using bundle base path for Neo4j: {APP_PATH}")
    else:
        APP_PATH = os.path.dirname(sys.executable)
        logger.info(f"Using executable directory for Neo4j: {APP_PATH}")
    
    # Define Neo4j paths
    NEO4J_PATH = os.path.join(APP_PATH, 'Neo4jDB')
    logger.info(f"Setting Neo4j path to: {NEO4J_PATH}")
    
    # Create Neo4j directory if it doesn't exist
    os.makedirs(NEO4J_PATH, exist_ok=True)
    logger.info(f"Created Neo4j directory: {NEO4J_PATH}")
    
    # Create Neo4j subdirectories
    os.makedirs(os.path.join(NEO4J_PATH, 'data'), exist_ok=True)
    os.makedirs(os.path.join(NEO4J_PATH, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(NEO4J_PATH, 'conf'), exist_ok=True)
    
    # Create a marker file to indicate this is a bundled Neo4j
    marker_path = os.path.join(NEO4J_PATH, '.bundled')
    with open(marker_path, 'w') as f:
        f.write('This directory contains a Neo4j database bundled with the application.')
    
    # Create a marker file to preserve data between runs
    preserve_path = os.path.join(NEO4J_PATH, '.preserve')
    with open(preserve_path, 'w') as f:
        f.write('This file indicates that Neo4j data should be preserved between application runs.')
    
    # Set environment variables for Neo4j
    os.environ['NEO4J_BUNDLED'] = 'True'
    os.environ['NEO4J_DB_PATH'] = NEO4J_PATH
    
    # Add JRE directory for Neo4j
    JRE_PATH = os.path.join(APP_PATH, 'jre')
    os.makedirs(JRE_PATH, exist_ok=True)
    os.environ['NEO4J_JRE_PATH'] = JRE_PATH
    
    logger.info(f"Neo4j environment setup complete")
