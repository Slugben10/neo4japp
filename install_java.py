#!/usr/bin/env python3
"""
Neo4j Java 11 JRE Installer Script
This script downloads and installs a compatible Java 11 JRE for Neo4j.
"""

import os
import sys
import platform
import subprocess
import requests
import tarfile
import zipfile
import shutil
import tempfile

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_info(message):
    print(f"{Colors.BLUE}[INFO] {message}{Colors.ENDC}")

def log_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.ENDC}")

def log_warning(message):
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.ENDC}")

def log_error(message):
    print(f"{Colors.RED}[ERROR] {message}{Colors.ENDC}")

def check_java_version():
    """Check if compatible Java version is already installed"""
    try:
        result = subprocess.run(
            ["java", "-version"], 
            capture_output=True, 
            text=True,
            check=False
        )
        
        version_output = result.stderr
        
        if "11" in version_output or "17" in version_output:
            log_success(f"Compatible Java already installed: {version_output.strip()}")
            return True
        else:
            log_warning(f"Incompatible Java version: {version_output.strip()}")
            return False
    except Exception:
        log_warning("Java not found in system path")
        return False

def get_app_path():
    """Get the application path"""
    if getattr(sys, 'frozen', False):
        # Running from compiled app
        return os.path.dirname(sys.executable)
    else:
        # Running from script
        return os.path.dirname(os.path.abspath(__file__))

def download_java_jre():
    """Download the Java 11 JRE appropriate for this platform"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Temporary directory for download
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Set download URL based on platform
        if system == "darwin":  # macOS
            if "arm" in machine or "aarch64" in machine:  # Apple Silicon
                url = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21%2B9/OpenJDK11U-jre_aarch64_mac_hotspot_11.0.21_9.tar.gz"
                filename = "openjdk-11-jre.tar.gz"
            else:  # Intel Mac
                url = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21%2B9/OpenJDK11U-jre_x64_mac_hotspot_11.0.21_9.tar.gz"
                filename = "openjdk-11-jre.tar.gz"
        elif system == "windows":  # Windows
            url = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21%2B9/OpenJDK11U-jre_x64_windows_hotspot_11.0.21_9.zip"
            filename = "openjdk-11-jre.zip"
        elif system == "linux":  # Linux
            if "arm" in machine or "aarch64" in machine:  # ARM
                url = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21%2B9/OpenJDK11U-jre_aarch64_linux_hotspot_11.0.21_9.tar.gz"
                filename = "openjdk-11-jre.tar.gz"
            else:  # x86_64
                url = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21%2B9/OpenJDK11U-jre_x64_linux_hotspot_11.0.21_9.tar.gz"
                filename = "openjdk-11-jre.tar.gz"
        else:
            log_error(f"Unsupported platform: {system} {machine}")
            return None
        
        download_path = os.path.join(tmp_dir, filename)
        
        log_info(f"Downloading Java 11 JRE from {url}")
        log_info(f"This may take a few minutes depending on your internet connection...")
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            log_error(f"Failed to download Java JRE: HTTP {response.status_code}")
            return None
        
        # Save the downloaded file with progress indicator
        file_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        percent = int(100 * downloaded / file_size)
                        sys.stdout.write(f"\rDownloading: {percent}% [{downloaded} / {file_size} bytes]")
                        sys.stdout.flush()
        
        print()  # New line after progress
        log_success(f"Download complete: {download_path}")
        return download_path
    
    except Exception as e:
        log_error(f"Error downloading Java JRE: {str(e)}")
        return None

def install_java_jre(download_path):
    """Extract and install the Java JRE"""
    app_path = get_app_path()
    java_dir = os.path.join(app_path, "jre")
    
    # Create Java directory
    os.makedirs(java_dir, exist_ok=True)
    
    try:
        system = platform.system().lower()
        
        log_info(f"Extracting Java JRE to {java_dir}")
        
        # Extract based on file type
        if download_path.endswith(".tar.gz"):
            with tarfile.open(download_path, 'r:gz') as tar:
                # Extract to a temporary directory first
                tmp_extract = tempfile.mkdtemp()
                tar.extractall(tmp_extract)
                
                # Find the JRE directory (usually there's a single top dir)
                extracted_dirs = [d for d in os.listdir(tmp_extract) if os.path.isdir(os.path.join(tmp_extract, d))]
                if not extracted_dirs:
                    log_error("No directories found in extracted archive")
                    return False
                
                # Move the contents to the Java directory
                src_dir = os.path.join(tmp_extract, extracted_dirs[0])
                for item in os.listdir(src_dir):
                    s = os.path.join(src_dir, item)
                    d = os.path.join(java_dir, item)
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)
                
                # Clean up
                shutil.rmtree(tmp_extract)
                
        elif download_path.endswith(".zip"):
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                # Extract to a temporary directory first
                tmp_extract = tempfile.mkdtemp()
                zip_ref.extractall(tmp_extract)
                
                # Find the JRE directory
                extracted_dirs = [d for d in os.listdir(tmp_extract) if os.path.isdir(os.path.join(tmp_extract, d))]
                if not extracted_dirs:
                    log_error("No directories found in extracted archive")
                    return False
                
                # Move the contents to the Java directory
                src_dir = os.path.join(tmp_extract, extracted_dirs[0])
                for item in os.listdir(src_dir):
                    s = os.path.join(src_dir, item)
                    d = os.path.join(java_dir, item)
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)
                
                # Clean up
                shutil.rmtree(tmp_extract)
        else:
            log_error(f"Unsupported file format: {download_path}")
            return False
        
        # Verify java executable exists
        java_exe = os.path.join(java_dir, "bin", "java")
        if system == "windows":
            java_exe += ".exe"
        
        if not os.path.exists(java_exe):
            log_error(f"Java executable not found at {java_exe}")
            return False
        
        # Make all executables in bin directory executable on Unix-like systems
        if system != "windows":
            bin_dir = os.path.join(java_dir, "bin")
            for file in os.listdir(bin_dir):
                file_path = os.path.join(bin_dir, file)
                if os.path.isfile(file_path):
                    # Make executable
                    current_mode = os.stat(file_path).st_mode
                    os.chmod(file_path, current_mode | 0o111)  # Add execute permission for all
        
        log_success(f"Java 11 JRE installed successfully to {java_dir}")
        return True
    
    except Exception as e:
        log_error(f"Error installing Java JRE: {str(e)}")
        return False

def configure_neo4j_for_java():
    """Configure Neo4j to use the installed Java"""
    app_path = get_app_path()
    jre_dir = os.path.join(app_path, "jre")
    neo4j_dir = os.path.join(app_path, "Neo4jDB", "neo4j-server")
    
    if not os.path.exists(neo4j_dir):
        log_warning(f"Neo4j installation not found at {neo4j_dir}")
        return False
    
    try:
        log_info("Configuring Neo4j to use the installed Java")
        
        # Determine Java bin path and home directory based on platform
        system = platform.system().lower()
        if system == "darwin":
            # Check for macOS JRE structure
            macos_java_bin = os.path.join(jre_dir, "Contents", "Home", "bin", "java")
            if os.path.exists(macos_java_bin):
                java_bin = macos_java_bin
                java_home = os.path.join(jre_dir, "Contents", "Home")
            else:
                java_bin = os.path.join(jre_dir, "bin", "java")
                java_home = jre_dir
        else:
            java_bin = os.path.join(jre_dir, "bin", "java")
            java_home = jre_dir
            if system == "windows":
                java_bin += ".exe"
        
        # Verify the Java executable exists
        if not os.path.exists(java_bin):
            log_error(f"Java executable not found at {java_bin}")
            return False
                
        # Set environment variables
        os.environ["JAVA_HOME"] = java_home
        path_separator = ";" if system == "windows" else ":"
        os.environ["PATH"] = os.path.dirname(java_bin) + path_separator + os.environ.get("PATH", "")
        
        # Find the wrapper configuration file
        wrapper_conf_path = os.path.join(neo4j_dir, "conf", "neo4j-wrapper.conf")
        
        # Create conf directory if it doesn't exist
        os.makedirs(os.path.dirname(wrapper_conf_path), exist_ok=True)
        
        if not os.path.exists(wrapper_conf_path):
            # Create the file if it doesn't exist
            with open(wrapper_conf_path, 'w') as f:
                f.write("# Neo4j wrapper configuration\n")
        
        # Add Java path to wrapper configuration
        if system == "windows":
            java_bin_esc = java_bin.replace("\\", "\\\\")  # Escape backslashes for Windows
        else:
            java_bin_esc = java_bin
        
        with open(wrapper_conf_path, 'w') as f:
            f.write("# Neo4j wrapper configuration\n")
            f.write("\n# Custom Java path configuration\n")
            f.write(f"wrapper.java.command={java_bin_esc}\n")
        
        # For Unix-like systems, also patch the Neo4j startup script
        if system != "windows":
            neo4j_script = os.path.join(neo4j_dir, "bin", "neo4j")
            if os.path.exists(neo4j_script):
                try:
                    # Read the script
                    with open(neo4j_script, 'r') as f:
                        script_lines = f.readlines()
                    
                    # Look for a good place to insert our JAVA_HOME definition
                    # Usually after the shebang line
                    insert_pos = 1  # Default to after shebang
                    for i, line in enumerate(script_lines):
                        if i == 0:  # Skip the shebang line
                            continue
                        if not line.strip().startswith('#'):  # First non-comment line
                            insert_pos = i
                            break
                    
                    # Insert JAVA_HOME and PATH export at the appropriate position
                    script_lines.insert(insert_pos, f'export JAVA_HOME="{java_home}"\n')
                    script_lines.insert(insert_pos + 1, f'export PATH="{os.path.dirname(java_bin)}:$PATH"\n')
                    
                    # Write the updated script
                    with open(neo4j_script, 'w') as f:
                        f.writelines(script_lines)
                    
                    # Make the script executable
                    os.chmod(neo4j_script, 0o755)
                    
                except Exception as e:
                    log_warning(f"Could not update Neo4j startup script: {e}")
        else:
            # For Windows, update the batch file
            neo4j_bat = os.path.join(neo4j_dir, "bin", "neo4j.bat")
            if os.path.exists(neo4j_bat):
                try:
                    # Read the script
                    with open(neo4j_bat, 'r') as f:
                        script_lines = f.readlines()
                    
                    # Look for a good place to insert our JAVA_HOME definition
                    insert_pos = 0
                    for i, line in enumerate(script_lines):
                        if line.lower().startswith('rem'):  # Skip REM comments
                            continue
                        insert_pos = i
                        break
                    
                    # Insert JAVA_HOME and PATH settings
                    script_lines.insert(insert_pos, f'set "JAVA_HOME={java_home}"\r\n')
                    script_lines.insert(insert_pos + 1, f'set "PATH={os.path.dirname(java_bin)};%PATH%"\r\n')
                    
                    # Write the updated script
                    with open(neo4j_bat, 'w') as f:
                        f.writelines(script_lines)
                    
                except Exception as e:
                    log_warning(f"Could not update Neo4j batch file: {e}")
        
        log_success(f"Neo4j configured to use Java at {java_bin}")
        return True
    
    except Exception as e:
        log_error(f"Error configuring Neo4j for Java: {str(e)}")
        return False

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}Neo4j Java 11 JRE Installer{Colors.ENDC}\n")
    
    # Check if Java 11+ is already installed
    if check_java_version():
        response = input("\nA compatible version of Java is already installed. Install anyway? (y/n): ")
        if response.lower() != 'y':
            log_info("Installation skipped. Using system Java.")
            return
    
    # Download Java JRE
    download_path = download_java_jre()
    if not download_path:
        log_error("Failed to download Java JRE. Installation aborted.")
        return
    
    # Install Java JRE
    if not install_java_jre(download_path):
        log_error("Failed to install Java JRE. Installation aborted.")
        return
    
    # Configure Neo4j
    if not configure_neo4j_for_java():
        log_warning("Failed to configure Neo4j to use the installed Java.")
        print(f"\n{Colors.YELLOW}Please restart the application for the Java installation to take effect.{Colors.ENDC}")
        return
    
    # Clean up the downloaded file
    try:
        os.remove(download_path)
        shutil.rmtree(os.path.dirname(download_path))
    except Exception:
        pass
    
    log_success("Java 11 JRE installation complete!")
    print(f"\n{Colors.GREEN}Please restart the application for the Java installation to take effect.{Colors.ENDC}")

if __name__ == "__main__":
    main() 