#!/usr/bin/env python3
import os
import sys
import platform
import time
import subprocess
import requests
import zipfile
import tarfile
import shutil
import traceback
import atexit
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("neo4j_downloader")

# Constants
NEO4J_VERSION = "4.4.30"  # Using a stable LTS version
NEO4J_PORT = 7687  # Default Bolt port

# Determine app path - handle macOS app bundles properly
def get_app_path():
    # Check if we're running from a frozen bundle (PyInstaller)
    if getattr(sys, 'frozen', False):
        if platform.system() == 'Darwin':  # macOS
            # Check for environment variable set by the app
            app_path = os.environ.get('RA_APP_PATH')
            if app_path and os.path.exists(app_path):
                logger.info(f"Using app path from environment: {app_path}")
                return app_path
                
            # Check various possible locations
            executable_path = os.path.dirname(sys.executable)
            
            # Check if we're in Contents/MacOS
            if 'Contents/MacOS' in executable_path:
                # Go up to Resources directory which should contain our data
                resources_path = os.path.abspath(os.path.join(executable_path, '../Resources'))
                if os.path.exists(resources_path):
                    logger.info(f"Using Resources path: {resources_path}")
                    return resources_path
                    
                # If Resources doesn't exist, go up to app bundle root
                app_bundle_root = os.path.abspath(os.path.join(executable_path, '../..'))
                logger.info(f"Using app bundle root: {app_bundle_root}")
                return app_bundle_root
        
        # For Windows/Linux or macOS fallback
        app_path = os.path.dirname(sys.executable)
    else:
        # Running as a script
        app_path = os.path.dirname(os.path.abspath(__file__))
        
    logger.info(f"Determined app path: {app_path}")
    return app_path

APP_PATH = get_app_path()

def log_message(message, is_error=False):
    """Simple logging function"""
    if is_error:
        logger.error(message)
    else:
        logger.info(message)

class EmbeddedNeo4jServer:
    def __init__(self, base_path=None):
        self.base_path = base_path or APP_PATH
        log_message(f"Using base path: {self.base_path}")
        
        # Create a directory structure that works with both app bundles and standard installs
        self.neo4j_dir = os.path.join(self.base_path, "Neo4jDB")
        self.server_dir = os.path.join(self.neo4j_dir, "neo4j-server")
        self.data_dir = os.path.join(self.neo4j_dir, "data")
        self.logs_dir = os.path.join(self.neo4j_dir, "logs")
        self.conf_dir = os.path.join(self.neo4j_dir, "conf")
        self.plugins_dir = os.path.join(self.neo4j_dir, "plugins")
        self.import_dir = os.path.join(self.neo4j_dir, "import")
        self.process = None
        self.running = False
        
        # Create all necessary directories
        dirs_to_create = [
            self.neo4j_dir,
            self.server_dir,
            self.data_dir,
            self.logs_dir,
            self.conf_dir,
            self.plugins_dir,
            self.import_dir
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            log_message(f"Created directory: {directory}")
        
        # Create marker files for the app to detect
        self._create_marker_files()
        
    def _create_marker_files(self):
        """Create marker files to indicate Neo4j is bundled and data should be preserved"""
        bundled_marker = os.path.join(self.neo4j_dir, '.bundled')
        with open(bundled_marker, 'w') as f:
            f.write('This directory contains a Neo4j database bundled with the application.\n')
            f.write(f'Created at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        preserve_marker = os.path.join(self.neo4j_dir, '.preserve')
        with open(preserve_marker, 'w') as f:
            f.write('This file indicates that Neo4j data should be preserved between application runs.\n')
            f.write('Delete this file if you want to reset the database on next startup.\n')
            f.write(f'Created at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        log_message("Created Neo4j marker files")
        
    def download_if_needed(self):
        """Download Neo4j server if it doesn't exist"""
        try:
            # Check if Neo4j is already installed
            if self._is_neo4j_installed():
                log_message(f"Neo4j server already installed at {self.server_dir}")
                
                # Still ensure it's configured correctly
                self._configure_neo4j()
                return True
                
            log_message(f"Downloading Neo4j {NEO4J_VERSION}...")
            
            # Determine download URL based on system
            system = platform.system().lower()
            if system == "windows":
                url = f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-windows.zip"
                archive_path = os.path.join(self.neo4j_dir, "neo4j.zip")
            else:  # Linux or macOS
                url = f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-unix.tar.gz"
                archive_path = os.path.join(self.neo4j_dir, "neo4j.tar.gz")
            
            # Download the archive
            log_message(f"Downloading from {url}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for error status codes
            except requests.exceptions.RequestException as e:
                log_message(f"Failed to download Neo4j: {str(e)}", True)
                return False
                
            # Save the downloaded file
            log_message(f"Saving to {archive_path}")
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            log_message(f"Downloaded Neo4j to {archive_path}")
            
            # Extract the archive
            log_message("Extracting archive...")
            extraction_dir = os.path.join(self.neo4j_dir, "temp_extract")
            os.makedirs(extraction_dir, exist_ok=True)
            
            try:
                if system == "windows":
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(extraction_dir)
                else:
                    with tarfile.open(archive_path, 'r:gz') as tar_ref:
                        tar_ref.extractall(extraction_dir)
                        
                # Find the extracted directory
                extracted_dirs = [d for d in os.listdir(extraction_dir) if os.path.isdir(os.path.join(extraction_dir, d))]
                if not extracted_dirs:
                    log_message("Extraction completed but no directories found", True)
                    return False
                    
                extracted_dir = os.path.join(extraction_dir, extracted_dirs[0])
                log_message(f"Extracted to: {extracted_dir}")
                
                # Move the extracted directory to server_dir
                if os.path.exists(self.server_dir) and os.listdir(self.server_dir):
                    log_message(f"Removing existing server directory: {self.server_dir}")
                    shutil.rmtree(self.server_dir)
                    
                log_message(f"Moving {extracted_dir} to {self.server_dir}")
                shutil.move(extracted_dir, self.server_dir)
                
                # Clean up
                shutil.rmtree(extraction_dir, ignore_errors=True)
                os.remove(archive_path)
                
            except Exception as e:
                log_message(f"Error during extraction: {str(e)}", True)
                log_message(traceback.format_exc(), True)
                return False
            
            # Configure Neo4j
            self._configure_neo4j()
            
            # Make scripts executable on Unix-like systems
            if system != "windows":
                bin_dir = os.path.join(self.server_dir, "bin")
                for script in os.listdir(bin_dir):
                    script_path = os.path.join(bin_dir, script)
                    if os.path.isfile(script_path) and not script.endswith(".bat"):
                        os.chmod(script_path, 0o755)
                        log_message(f"Made executable: {script_path}")
            
            log_message("Neo4j server installed successfully")
            return True
            
        except Exception as e:
            log_message(f"Error downloading Neo4j: {str(e)}", True)
            log_message(traceback.format_exc(), True)
            return False
    
    def _configure_neo4j(self):
        """Configure Neo4j settings"""
        try:
            config_path = os.path.join(self.server_dir, "conf", "neo4j.conf")
            if not os.path.exists(config_path):
                log_message(f"Neo4j config file not found at {config_path}", True)
                
                # Try to create a simple config file
                config_dir = os.path.dirname(config_path)
                os.makedirs(config_dir, exist_ok=True)
                
                with open(config_path, 'w') as f:
                    f.write(f"# Auto-generated Neo4j configuration\n")
                    f.write(f"dbms.directories.data={self.data_dir}\n")
                    f.write(f"dbms.directories.logs={self.logs_dir}\n")
                    f.write(f"dbms.directories.plugins={self.plugins_dir}\n")
                    f.write(f"dbms.directories.import={self.import_dir}\n")
                    f.write(f"dbms.security.procedures.unrestricted=apoc.*\n")
                    f.write(f"dbms.security.auth_enabled=false\n")
                    f.write(f"dbms.tx_state.memory_allocation=ON_HEAP\n")
                    f.write(f"dbms.connectors.default_listen_address=localhost\n")
                    f.write(f"dbms.connector.bolt.listen_address=localhost:7687\n")
                    
                log_message(f"Created new Neo4j config file at {config_path}")
                return True
            
            # Read the configuration file
            with open(config_path, 'r') as f:
                config_lines = f.readlines()
            
            # Modify the configuration
            new_config_lines = []
            settings_applied = {
                "dbms.directories.data": False,
                "dbms.directories.logs": False,
                "dbms.directories.plugins": False,
                "dbms.directories.import": False,
                "dbms.security.procedures.unrestricted": False,
                "dbms.security.auth_enabled": False,
                "dbms.tx_state.memory_allocation": False,
                "dbms.connectors.default_listen_address": False,
                "dbms.connector.bolt.listen_address": False,
            }
            
            for line in config_lines:
                line_stripped = line.strip()
                
                # Set data directory
                if line_stripped.startswith('#dbms.directories.data=') or line_stripped.startswith('dbms.directories.data='):
                    line = f"dbms.directories.data={self.data_dir}\n"
                    settings_applied["dbms.directories.data"] = True
                    
                # Set logs directory
                elif line_stripped.startswith('#dbms.directories.logs=') or line_stripped.startswith('dbms.directories.logs='):
                    line = f"dbms.directories.logs={self.logs_dir}\n"
                    settings_applied["dbms.directories.logs"] = True
                    
                # Set plugins directory
                elif line_stripped.startswith('#dbms.directories.plugins=') or line_stripped.startswith('dbms.directories.plugins='):
                    line = f"dbms.directories.plugins={self.plugins_dir}\n"
                    settings_applied["dbms.directories.plugins"] = True
                    
                # Set import directory
                elif line_stripped.startswith('#dbms.directories.import=') or line_stripped.startswith('dbms.directories.import='):
                    line = f"dbms.directories.import={self.import_dir}\n"
                    settings_applied["dbms.directories.import"] = True
                    
                # Enable APOC
                elif line_stripped.startswith('#dbms.security.procedures.unrestricted=') or line_stripped.startswith('dbms.security.procedures.unrestricted='):
                    line = "dbms.security.procedures.unrestricted=apoc.*\n"
                    settings_applied["dbms.security.procedures.unrestricted"] = True
                    
                # Disable authentication for local use
                elif line_stripped.startswith('#dbms.security.auth_enabled=') or line_stripped.startswith('dbms.security.auth_enabled='):
                    line = "dbms.security.auth_enabled=false\n"
                    settings_applied["dbms.security.auth_enabled"] = True
                    
                # Set memory allocation
                elif line_stripped.startswith('#dbms.tx_state.memory_allocation=') or line_stripped.startswith('dbms.tx_state.memory_allocation='):
                    line = "dbms.tx_state.memory_allocation=ON_HEAP\n"
                    settings_applied["dbms.tx_state.memory_allocation"] = True
                    
                # Set default listen address
                elif line_stripped.startswith('#dbms.connectors.default_listen_address=') or line_stripped.startswith('dbms.connectors.default_listen_address='):
                    line = "dbms.connectors.default_listen_address=localhost\n"
                    settings_applied["dbms.connectors.default_listen_address"] = True
                    
                # Set Bolt listen address
                elif line_stripped.startswith('#dbms.connector.bolt.listen_address=') or line_stripped.startswith('dbms.connector.bolt.listen_address='):
                    line = "dbms.connector.bolt.listen_address=localhost:7687\n"
                    settings_applied["dbms.connector.bolt.listen_address"] = True
                    
                new_config_lines.append(line)
            
            # Add any missing settings at the end
            for setting, applied in settings_applied.items():
                if not applied:
                    if setting == "dbms.directories.data":
                        new_config_lines.append(f"dbms.directories.data={self.data_dir}\n")
                    elif setting == "dbms.directories.logs":
                        new_config_lines.append(f"dbms.directories.logs={self.logs_dir}\n")
                    elif setting == "dbms.directories.plugins":
                        new_config_lines.append(f"dbms.directories.plugins={self.plugins_dir}\n")
                    elif setting == "dbms.directories.import":
                        new_config_lines.append(f"dbms.directories.import={self.import_dir}\n")
                    elif setting == "dbms.security.procedures.unrestricted":
                        new_config_lines.append("dbms.security.procedures.unrestricted=apoc.*\n")
                    elif setting == "dbms.security.auth_enabled":
                        new_config_lines.append("dbms.security.auth_enabled=false\n")
                    elif setting == "dbms.tx_state.memory_allocation":
                        new_config_lines.append("dbms.tx_state.memory_allocation=ON_HEAP\n")
                    elif setting == "dbms.connectors.default_listen_address":
                        new_config_lines.append("dbms.connectors.default_listen_address=localhost\n")
                    elif setting == "dbms.connector.bolt.listen_address":
                        new_config_lines.append("dbms.connector.bolt.listen_address=localhost:7687\n")
            
            # Write the modified configuration
            with open(config_path, 'w') as f:
                f.writelines(new_config_lines)
            
            log_message("Neo4j configured successfully")
            
            # Also copy config to our app-specific config directory for backup
            backup_config = os.path.join(self.conf_dir, "neo4j.conf")
            shutil.copy2(config_path, backup_config)
            log_message(f"Config backed up to {backup_config}")
            
            return True
        except Exception as e:
            log_message(f"Error configuring Neo4j: {str(e)}", True)
            log_message(traceback.format_exc(), True)
            return False
    
    def _is_neo4j_installed(self):
        """Check if Neo4j is already installed"""
        # Check for bin directory with neo4j executable
        if platform.system().lower() == "windows":
            neo4j_executable = os.path.join(self.server_dir, "bin", "neo4j.bat")
        else:
            neo4j_executable = os.path.join(self.server_dir, "bin", "neo4j")
            
        if os.path.exists(neo4j_executable):
            log_message(f"Found Neo4j executable at {neo4j_executable}")
            return True
            
        log_message(f"Neo4j executable not found at {neo4j_executable}")
        return False

def main():
    print("=" * 60)
    print("Neo4j Downloader for Research Assistant")
    print("=" * 60)
    print(f"App path: {APP_PATH}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("-" * 60)
    
    # Create Neo4j server instance
    server = EmbeddedNeo4jServer(APP_PATH)
    
    print(f"Downloading Neo4j {NEO4J_VERSION}...")
    if server.download_if_needed():
        print("=" * 60)
        print("Neo4j server downloaded and configured successfully!")
        print(f"Server directory: {server.server_dir}")
        print(f"Data directory: {server.data_dir}")
        print(f"Logs directory: {server.logs_dir}")
        print(f"Config directory: {server.conf_dir}")
        print("-" * 60)
        print("You can now start the application, which will automatically use this Neo4j server.")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("Failed to download or configure Neo4j server.")
        print("Please check your internet connection and try again.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 