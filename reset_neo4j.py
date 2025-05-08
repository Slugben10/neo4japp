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

def reset_neo4j_database():
    """Reset Neo4j database by removing data files"""
    app_path = get_app_path()
    neo4j_dir = os.path.join(app_path, "Neo4jDB", "neo4j-server")
    data_dir = os.path.join(app_path, "Neo4jDB", "data")
    
    # Make sure the data directory exists
    if not os.path.exists(data_dir):
        log_message(f"Neo4j data directory not found at {data_dir}")
        return False
    
    try:
        # Remove data directory
        log_message(f"Removing Neo4j data directory at {data_dir}")
        shutil.rmtree(data_dir)
        log_message("Neo4j data directory removed successfully")
        
        # Recreate empty data directory
        os.makedirs(data_dir, exist_ok=True)
        log_message("Created empty Neo4j data directory")
        
        # Remove .preserve file if it exists
        preserve_file = os.path.join(app_path, "Neo4jDB", ".preserve")
        if os.path.exists(preserve_file):
            os.remove(preserve_file)
            log_message("Removed Neo4j data preservation marker")
        
        # Create a new .preserve file after reset to ensure data isn't reset again
        with open(preserve_file, 'w') as f:
            f.write("# This file indicates that Neo4j data should be preserved\n")
        log_message("Created new Neo4j data preservation marker")
        
        return True
    except Exception as e:
        log_message(f"Error resetting Neo4j database: {str(e)}")
        return False

def main():
    print("\n=== Neo4j Database Reset ===\n")
    log_message("This script will completely reset your Neo4j database.")
    log_message("WARNING: All data in the database will be permanently deleted!")
    
    # Ask for confirmation
    confirm = input("\nAre you sure you want to reset the Neo4j database? (yes/no): ")
    if confirm.lower() != "yes":
        print("Reset cancelled.")
        return 0
    
    # Reset Neo4j database
    if not reset_neo4j_database():
        print("[ERROR] Failed to reset Neo4j database.")
        return 1
    
    print("\n[SUCCESS] Neo4j database has been reset.")
    print("[INFO] Please restart your application for changes to take effect.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 