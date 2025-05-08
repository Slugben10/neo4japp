
# General application hook
import os
import sys
import json

# Ensure we can find the app's resources
if getattr(sys, 'frozen', False):
    # Running as a bundled executable
    APP_PATH = os.path.dirname(sys.executable)
    os.environ['RA_APP_PATH'] = APP_PATH
    
    # Load configuration to check Neo4j preservation settings
    config_path = os.path.join(APP_PATH, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set environment variables for Neo4j preservation
            if config.get('preserve_neo4j_data', True):
                os.environ['PRESERVE_NEO4J_DATA'] = 'True'
            else:
                os.environ['PRESERVE_NEO4J_DATA'] = 'False'
                
            if config.get('download_neo4j_if_missing', True):
                os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True'
            else:
                os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'False'
        except Exception:
            # Default to preserving data if config can't be loaded
            os.environ['PRESERVE_NEO4J_DATA'] = 'True'
            os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True'
    else:
        # Default to preserving data if config doesn't exist
        os.environ['PRESERVE_NEO4J_DATA'] = 'True'
        os.environ['DOWNLOAD_NEO4J_IF_MISSING'] = 'True'
