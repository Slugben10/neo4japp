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
from pathlib import Path

# Constants
NEO4J_VERSION = "4.4.30"  # Using a stable LTS version
NEO4J_PORT = 7687  # Default Bolt port
APP_PATH = os.path.dirname(os.path.abspath(__file__))

def log_message(message, is_error=False):
    """Simple logging function"""
    prefix = "[ERROR]" if is_error else "[INFO]"
    print(f"{prefix} {message}")

class EmbeddedNeo4jServer:
    def __init__(self, base_path=None):
        self.base_path = base_path or APP_PATH
        self.server_dir = os.path.join(self.base_path, "Neo4jDB", "neo4j-server")
        self.data_dir = os.path.join(self.base_path, "Neo4jDB", "data")
        self.logs_dir = os.path.join(self.base_path, "Neo4jDB", "logs")
        self.process = None
        self.running = False
        
        # Create directories
        os.makedirs(self.server_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def download_if_needed(self):
        """Download Neo4j server if it doesn't exist"""
        try:
            # Check if Neo4j is already installed
            if self._is_neo4j_installed():
                log_message(f"Neo4j server already installed at {self.server_dir}")
                return True
                
            log_message(f"Downloading Neo4j {NEO4J_VERSION}...")
            
            # Determine download URL based on system
            system = platform.system().lower()
            if system == "windows":
                url = f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-windows.zip"
                archive_path = os.path.join(self.base_path, "Neo4jDB", "neo4j.zip")
            else:  # Linux or macOS
                url = f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-unix.tar.gz"
                archive_path = os.path.join(self.base_path, "Neo4jDB", "neo4j.tar.gz")
            
            # Download the archive
            log_message(f"Downloading from {url}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                log_message(f"Failed to download Neo4j: HTTP {response.status_code}", True)
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
            if system == "windows":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(self.base_path, "Neo4jDB"))
            else:
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(os.path.join(self.base_path, "Neo4jDB"))
            
            # Rename the extracted directory
            extracted_dir = os.path.join(self.base_path, "Neo4jDB", f"neo4j-community-{NEO4J_VERSION}")
            if os.path.exists(extracted_dir):
                if os.path.exists(self.server_dir):
                    shutil.rmtree(self.server_dir)
                shutil.move(extracted_dir, self.server_dir)
            
            # Remove the archive
            os.remove(archive_path)
            
            # Configure Neo4j
            self._configure_neo4j()
            
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
                return False
            
            # Read the configuration file
            with open(config_path, 'r') as f:
                config_lines = f.readlines()
            
            # Modify the configuration
            new_config_lines = []
            for line in config_lines:
                # Set data directory
                if line.strip().startswith('#dbms.directories.data='):
                    line = f"dbms.directories.data={self.data_dir}\n"
                # Set logs directory
                elif line.strip().startswith('#dbms.directories.logs='):
                    line = f"dbms.directories.logs={self.logs_dir}\n"
                # Enable APOC
                elif line.strip().startswith('#dbms.security.procedures.unrestricted='):
                    line = "dbms.security.procedures.unrestricted=apoc.*\n"
                # Disable authentication for local use
                elif line.strip().startswith('#dbms.security.auth_enabled='):
                    line = "dbms.security.auth_enabled=false\n"
                new_config_lines.append(line)
            
            # Write the modified configuration
            with open(config_path, 'w') as f:
                f.writelines(new_config_lines)
            
            log_message("Neo4j configured successfully")
            return True
        except Exception as e:
            log_message(f"Error configuring Neo4j: {str(e)}", True)
            log_message(traceback.format_exc(), True)
            return False
    
    def _is_neo4j_installed(self):
        """Check if Neo4j is already installed"""
        # Check for bin directory with neo4j executable
        if platform.system().lower() == "windows":
            return os.path.exists(os.path.join(self.server_dir, "bin", "neo4j.bat"))
        else:
            return os.path.exists(os.path.join(self.server_dir, "bin", "neo4j"))

def main():
    print("Neo4j Downloader for Research Assistant")
    print("---------------------------------------")
    print(f"App path: {APP_PATH}")
    
    # Create Neo4j server instance
    server = EmbeddedNeo4jServer(APP_PATH)
    
    print(f"Downloading Neo4j {NEO4J_VERSION}...")
    if server.download_if_needed():
        print("Neo4j server downloaded and configured successfully!")
        print(f"Server directory: {server.server_dir}")
        print(f"Data directory: {server.data_dir}")
        print(f"Logs directory: {server.logs_dir}")
        return 0
    else:
        print("Failed to download or configure Neo4j server.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 