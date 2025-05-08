
# Neo4j bundling hook
import os
import sys
import shutil

# If we're frozen (in the final executable)
if getattr(sys, 'frozen', False):
    APP_PATH = os.path.dirname(sys.executable)
    
    # Define Neo4j paths
    NEO4J_PATH = os.path.join(APP_PATH, 'Neo4jDB')
    
    # Create Neo4j directory if it doesn't exist
    os.makedirs(NEO4J_PATH, exist_ok=True)
    
    # Create a marker file to indicate this is a bundled Neo4j
    marker_path = os.path.join(NEO4J_PATH, '.bundled')
    with open(marker_path, 'w') as f:
        f.write('This directory contains a Neo4j database bundled with the application.')
        
    # Set environment variable to indicate Neo4j is bundled
    os.environ['NEO4J_BUNDLED'] = 'True'
    os.environ['NEO4J_DB_PATH'] = NEO4J_PATH
