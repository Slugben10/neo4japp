#!/usr/bin/env python3
import os
import sys
import shutil
import glob

def log_message(message):
    print(f"[INFO] {message}")

def get_app_path():
    """Get the application path"""
    if getattr(sys, 'frozen', False):
        # Running as compiled app
        return os.path.dirname(sys.executable)
    else:
        # Running from script
        return os.path.dirname(os.path.abspath(__file__))

def fix_neo4j_config():
    """Fix Neo4j configuration to disable shell server"""
    app_path = get_app_path()
    neo4j_dir = os.path.join(app_path, "Neo4jDB", "neo4j-server")
    config_path = os.path.join(neo4j_dir, "conf", "neo4j.conf")
    
    if not os.path.exists(config_path):
        log_message(f"Neo4j config file not found at {config_path}")
        return False
    
    # Backup the config file
    backup_path = f"{config_path}.bak"
    shutil.copy2(config_path, backup_path)
    log_message(f"Created backup of Neo4j config file at {backup_path}")
    
    # Read the configuration file
    with open(config_path, 'r') as f:
        config_lines = f.readlines()
    
    # Check if we've already fixed this file
    already_fixed = any("dbms.shell.enabled=false" in line for line in config_lines)
    if already_fixed:
        log_message("Neo4j config already has shell server disabled")
        return True
    
    # Modify the configuration
    new_config_lines = []
    modified = False
    
    for line in config_lines:
        # Disable shell server to avoid port conflicts (1337) commonly seen on Mac
        if line.strip().startswith('#dbms.shell.enabled='):
            line = "dbms.shell.enabled=false\n"
            modified = True
        new_config_lines.append(line)
    
    # If we didn't find the shell config line, add it explicitly
    if not modified:
        new_config_lines.append("\n# Disable shell server to prevent port conflicts\n")
        new_config_lines.append("dbms.shell.enabled=false\n")
    
    # Write the modified configuration
    with open(config_path, 'w') as f:
        f.writelines(new_config_lines)
    
    log_message("Neo4j config updated to disable shell server")
    return True

def cleanup_lock_files():
    """Remove Neo4j lock files to ensure clean restart"""
    app_path = get_app_path()
    data_dir = os.path.join(app_path, "Neo4jDB", "data")
    
    if not os.path.exists(data_dir):
        log_message("Neo4j data directory not found")
        return
    
    # Find and remove lock files
    lock_files = glob.glob(f"{data_dir}/**/*.lock", recursive=True)
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
            log_message(f"Removed lock file: {lock_file}")
        except Exception as e:
            log_message(f"Error removing lock file {lock_file}: {str(e)}")

def main():
    print("\n=== Neo4j Port Conflict Fix ===\n")
    log_message("This script will fix port conflicts by disabling Neo4j shell server")
    
    # Fix Neo4j configuration
    if not fix_neo4j_config():
        print("[ERROR] Failed to update Neo4j configuration.")
        return 1
    
    # Clean up lock files
    cleanup_lock_files()
    
    print("\n[SUCCESS] Neo4j configuration updated to fix port conflicts.")
    print("[INFO] Please restart your application for changes to take effect.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 