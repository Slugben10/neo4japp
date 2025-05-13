import os
import shutil
import sys
import json
import subprocess
import importlib.util

# Define the app name for consistent reference
APP_NAME = "RA"

# Ensure base directories exist
os.makedirs("Documents", exist_ok=True)
os.makedirs("Neo4jDB", exist_ok=True)
os.makedirs("Prompts", exist_ok=True)

print(f"Starting build process for {APP_NAME} with wxPython...")

# Check for required dependencies and install if missing
required_packages = [
    "altgraph", 
    "PyInstaller", 
    "neo4j", 
    "langchain>=0.0.240,<=0.0.312", 
    "langchain_core>=0.1.0", 
    "langchain_openai", 
    "langchain_neo4j", 
    "langchain_community", 
    "langchain_experimental",
    "openai",
    "pypdf",  # PDF processing
    "python-docx",  # DOCX processing
    # Google API dependencies - more specific packages
    "google-api-python-client",
    "google-api-core",
    "google-cloud-core",
    "google-cloud",
    "google-cloud-aiplatform",
    "google-cloud-storage",
    "google-generativeai>=0.3.0",  # Version constraint for newer features
    "protobuf>=4.23.0",  # Required by google packages
    "langchain-google-genai",  # Add hyphenated package name as fallback
    "langchain_google_genai",  # Add underscore package name
    # Anthropic client
    "anthropic",
    # Additional LangChain integrations
    "langchain_anthropic",
]
missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package.split('>=')[0].split('<=')[0])  # Handle version constraints in package names
        print(f"✓ {package} is installed")
    except ImportError:
        missing_packages.append(package)
        print(f"✗ {package} is missing")

if missing_packages:
    print(f"Installing missing dependencies: {', '.join(missing_packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + missing_packages)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Please install the following packages manually: " + ", ".join(missing_packages))
        sys.exit(1)

# Import PyInstaller after ensuring it's installed
import PyInstaller.__main__ as pyinstaller_main

# Try to import wx to get its path
try:
    import wx
    wx_path = os.path.dirname(wx.__file__)
    print(f"Found wxPython at: {wx_path}")
    wx_version = wx.__version__
    print(f"wxPython version: {wx_version}")
    HAS_WX = True
except ImportError:
    print("Warning: wxPython not found. Trying to continue anyway...")
    wx_path = ""
    HAS_WX = False

# Create a default config.json if it doesn't exist
if not os.path.exists("config.json"):
    print("Creating default config.json...")
    default_config = {
        "models": {
            "openai": {
                "name": "OpenAI GPT-4",
                "api_key_env": "OPENAI_API_KEY",
                "model_name": "gpt-4o-mini"
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model_name": "claude-3-7-sonnet-20250219"
            },
            "gemini": {
                "name": "Google Gemini-2.0-Flash",
                "api_key_env": "GOOGLE_API_KEY",
                "model_name": "gemini-2.0-flash"
            },
        },
        "default_model": "openai",
        "max_tokens": 8000,
        "system_prompt": "You are a helpful AI research assistant. Your goal is to help researchers write new papers or expand work-in-progress papers based on the provided documents and instructions.",
        "preserve_neo4j_data": True,  # Add flag to preserve Neo4j data between runs
        "download_neo4j_if_missing": True  # Add flag to download Neo4j if not found
    }
    
    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=2)
    print("Default config.json created successfully")
else:
    # Update existing config with Neo4j preservation settings if missing
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        updated = False
        if "preserve_neo4j_data" not in config:
            config["preserve_neo4j_data"] = True
            updated = True
            
        if "download_neo4j_if_missing" not in config:
            config["download_neo4j_if_missing"] = True
            updated = True
            
        if updated:
            with open("config.json", "w") as f:
                json.dump(config, f, indent=2)
            print("Updated config.json with Neo4j preservation settings")
    except Exception as e:
        print(f"Error updating config.json: {e}")

# Create runtime hooks directory if it doesn't exist
if not os.path.exists("hooks"):
    os.makedirs("hooks", exist_ok=True)

# Create wxPython hook for compatibility - simpler version to avoid altgraph issues
wx_hook_path = os.path.join("hooks", "hook-wx.py")
with open(wx_hook_path, 'w') as f:
    f.write("""
# wxPython hook for better compatibility with PyInstaller
hiddenimports = [
    'wx.lib.scrolledpanel',
    'wx.lib.newevent',
    'wx.lib.colourdb',
    'wx.adv',
    'wx.html',
    'wx.grid',
    'wx.lib.agw',
    'wx._xml',
    'wx._html',
    'wx._adv',
    'wx._core',
    'wx._controls',
]

# Platform-specific imports
import sys
if sys.platform == 'darwin':
    hiddenimports.extend(['wx.lib.osx'])
elif sys.platform == 'win32':
    hiddenimports.extend(['wx.msw'])
""")

# Create app initialization hook with improved path handling for app bundles
app_hook_path = os.path.join("hooks", "hook-app.py")
with open(app_hook_path, 'w') as f:
    f.write("""
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
""")

# Create Neo4j specific hook with improved path handling
neo4j_hook_path = os.path.join("hooks", "hook-neo4j-bundling.py")
with open(neo4j_hook_path, 'w') as f:
    f.write("""
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
""")

# Define data files to include
data_files = [
    ("config.json", "."),
    ("install_java.py", "."),  # Include Java installer script
    ("force_java_config.py", "."),  # Include Java force config script
    ("download_neo4j.py", "."),  # Include Neo4j downloader script
]

# Add .env file if it exists
if os.path.exists(".env"):
    data_files.append((".env", "."))

# Copy Neo4j files if they exist
neo4j_dir = os.path.join('Neo4jDB')
if os.path.exists(neo4j_dir) and os.path.isdir(neo4j_dir):
    # Add Neo4jDB directory structure - will be properly handled during build
    for root, dirs, files in os.walk(neo4j_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path)
            data_files.append((rel_path, os.path.dirname(rel_path)))
    print(f"Added Neo4j database files from {neo4j_dir}")

# Define hidden imports based on what's used in main.py
hidden_imports = [
    "wx",
    "wx.lib.scrolledpanel",
    "wx.lib.newevent",
    "json",
    "threading",
    "requests",
    "shutil",
    "traceback",
    "dotenv",
    "altgraph",  # Add altgraph explicitly
    "neo4j",
    "langchain",
    "langchain_core",
    "langchain_core.runnables",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    "langchain.text_splitter",
    "langchain.vectorstores",
    "langchain.schema",
    "langchain_openai",
    "langchain_neo4j",
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.neo4j_vector",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.pdf",
    "langchain_community.document_loaders.text",
    "langchain_community.document_loaders.docx",
    "langchain_experimental",
    "langchain_experimental.graph_transformers",
    # LLM provider-specific imports
    "openai",
    "anthropic",
    "google.generativeai",
    "google.api_core",
    "google.cloud",
    "google.cloud.aiplatform",
    "langchain_google_genai",
    "langchain_anthropic",
    # Add tiktoken and its extensions to fix "Unknown encoding cl100k_base" error
    "tiktoken",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
]

# Try to include optional packages
try:
    import docx
    hidden_imports.append("docx")
except ImportError:
    print("Warning: python-docx not installed. DOCX support will be limited.")

try:
    import pypdf
    hidden_imports.append("pypdf")
except ImportError:
    print("Warning: pypdf not installed. PDF support will be limited.")

# Base PyInstaller arguments
pyinstaller_args = [
    'main.py',
    '--name=' + APP_NAME,
    '--onedir',
    '--clean',
    '--noconfirm',
]

if sys.platform == 'darwin':  # Only hide console in macOS
    pyinstaller_args.append('--noconsole')  # Hide console window in macOS

# Add runtime hooks
pyinstaller_args.append('--runtime-hook=hooks/hook-app.py')
pyinstaller_args.append('--runtime-hook=hooks/hook-neo4j-bundling.py')

# Add additional hooks directory
pyinstaller_args.append('--additional-hooks-dir=hooks')

# Add hidden imports
for imp in hidden_imports:
    pyinstaller_args.append('--hidden-import=' + imp)

# Add data files
for src, dst in data_files:
    pyinstaller_args.append('--add-data=' + src + os.pathsep + dst)

# Platform specific settings
if sys.platform == 'darwin':  # macOS
    print("Building for macOS...")
    pyinstaller_args.append('--windowed')
    pyinstaller_args.append('--osx-bundle-identifier=com.researchassistant.app')
    
    # Add icon if available
    if os.path.exists('app_icon.icns'):
        pyinstaller_args.append('--icon=app_icon.icns')
elif sys.platform == 'win32':  # Windows
    print("Building for Windows...")
    
    # Add specific Windows options for wxPython
    pyinstaller_args.append('--hidden-import=wx.msw')
    
    # Add icon if available
    if os.path.exists('app_icon.ico'):
        pyinstaller_args.append('--icon=app_icon.ico')

# Add explicit tiktoken imports to fix "Unknown encoding cl100k_base" error
pyinstaller_args.append('--hidden-import=tiktoken')
pyinstaller_args.append('--hidden-import=tiktoken_ext')
pyinstaller_args.append('--hidden-import=tiktoken_ext.openai_public')

# Print the PyInstaller command for debugging
print("PyInstaller command:", " ".join(pyinstaller_args))

try:
    # Run PyInstaller
    pyinstaller_main.run(pyinstaller_args)

    # Set up the required directories in various locations
    dist_dir = os.path.join('dist', APP_NAME)
    
    # Ensure core directories exist in the output folder
    directories = ['Documents', 'Neo4jDB', 'Neo4jDB/data', 'Neo4jDB/logs', 'Prompts', 'jre']
    for directory in directories:
        dir_path = os.path.join(dist_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create a Neo4j preservation marker file
    neo4j_marker = os.path.join(dist_dir, 'Neo4jDB', '.preserve')
    with open(neo4j_marker, 'w') as f:
        f.write("This file indicates that Neo4j data should be preserved between application runs.\n")
        f.write("Delete this file if you want to reset the database on next startup.\n")
    
    # Create a JRE README
    jre_readme_path = os.path.join(dist_dir, 'jre', 'README.txt')
    with open(jre_readme_path, 'w') as f:
        f.write("This directory will contain a Java Runtime Environment (JRE) for Neo4j.\n")
        f.write("If you experience issues with Neo4j, the application will help you install Java here.\n")
        f.write("You can also run install_java.py manually to install Java.\n")

    # Copy key files to the output folder
    key_files = ['config.json', '.env', 'download_neo4j.py', 'install_java.py', 'force_java_config.py']
    for file in key_files:
        if os.path.exists(file):
            shutil.copy(file, dist_dir)
            print(f"Copied {file} to {dist_dir}")
            # Make executable if script
            if file.endswith('.py'):
                try:
                    if sys.platform != 'win32':  # Unix-like systems
                        os.chmod(os.path.join(dist_dir, file), 0o755)
                except Exception as e:
                    print(f"Warning: Could not make {file} executable: {e}")

    # For macOS, perform additional fixes for app bundle
    if sys.platform == 'darwin':
        app_bundle_path = os.path.join('dist', f"{APP_NAME}.app")
        if os.path.exists(app_bundle_path):
            print(f"Setting up macOS app bundle: {app_bundle_path}")
            
            # Get app bundle subdirectories
            contents_path = os.path.join(app_bundle_path, "Contents")
            macos_path = os.path.join(contents_path, "MacOS")
            resources_path = os.path.join(contents_path, "Resources")
            
            # Ensure core directories exist in both MacOS and Resources
            for base_dir in [macos_path, resources_path]:
                for directory in directories:
                    dir_path = os.path.join(base_dir, directory)
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory: {dir_path}")
                
                # Create Neo4j preservation marker
                neo4j_marker = os.path.join(base_dir, 'Neo4jDB', '.preserve')
                with open(neo4j_marker, 'w') as f:
                    f.write("This file indicates that Neo4j data should be preserved between application runs.\n")
                
                # Create JRE README
                jre_readme = os.path.join(base_dir, 'jre', 'README.txt')
                with open(jre_readme, 'w') as f:
                    f.write("This directory will contain a Java Runtime Environment (JRE) for Neo4j.\n")
                
                # Copy key files
                for file in key_files:
                    if os.path.exists(file):
                        try:
                            shutil.copy(file, base_dir)
                            print(f"Copied {file} to {base_dir}")
                            # Make executable if script
                            if file.endswith('.py'):
                                os.chmod(os.path.join(base_dir, file), 0o755)
                        except Exception as e:
                            print(f"Warning: Could not copy {file} to {base_dir}: {e}")
            
            # Create a fixed entry point script to resolve path issues
            entry_point_script = os.path.join(macos_path, f"{APP_NAME}_launcher.sh")
            with open(entry_point_script, 'w') as f:
                f.write(f"""#!/bin/bash
# Launcher script for {APP_NAME} that fixes path issues

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
APP_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
MACOS_DIR="$SCRIPT_DIR"
RESOURCES_DIR="$( cd "$SCRIPT_DIR/../Resources" && pwd )"

# Set up environment variables
export RA_APP_PATH="$RESOURCES_DIR"
export RA_BUNDLE_PATH="$APP_DIR"
export RA_RESOURCES_PATH="$RESOURCES_DIR"
export RA_MACOS_PATH="$MACOS_DIR"
export NEO4J_DB_PATH="$RESOURCES_DIR/Neo4jDB"
export NEO4J_BUNDLED="True"
export NEO4J_JRE_PATH="$RESOURCES_DIR/jre"
export PRESERVE_NEO4J_DATA="True"
export DOWNLOAD_NEO4J_IF_MISSING="True"

# Create necessary directories
mkdir -p "$RESOURCES_DIR/Neo4jDB/data"
mkdir -p "$RESOURCES_DIR/Neo4jDB/logs"
mkdir -p "$RESOURCES_DIR/Documents"
mkdir -p "$RESOURCES_DIR/Prompts"
mkdir -p "$RESOURCES_DIR/jre"

# Copy config.json if it exists in MACOS_DIR but not in RESOURCES_DIR
if [ -f "$MACOS_DIR/config.json" ] && [ ! -f "$RESOURCES_DIR/config.json" ]; then
    cp "$MACOS_DIR/config.json" "$RESOURCES_DIR/config.json"
fi

# Copy .env if it exists in MACOS_DIR but not in RESOURCES_DIR
if [ -f "$MACOS_DIR/.env" ] && [ ! -f "$RESOURCES_DIR/.env" ]; then
    cp "$MACOS_DIR/.env" "$RESOURCES_DIR/.env"
fi

# Launch the actual application
"$MACOS_DIR/{APP_NAME}" "$@"
""")
            
            # Make launcher script executable
            os.chmod(entry_point_script, 0o755)
            print(f"Created launcher script: {entry_point_script}")
            
            # Create a .command file that can be double-clicked to run the app in Terminal
            terminal_launcher = os.path.join('dist', f"{APP_NAME}_terminal.command")
            with open(terminal_launcher, 'w') as f:
                f.write(f"""#!/bin/bash
# Run {APP_NAME} with Terminal output for debugging

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
APP_PATH="$SCRIPT_DIR/{APP_NAME}.app/Contents/MacOS"

echo "Starting {APP_NAME} with Terminal output..."
"$APP_PATH/{APP_NAME}" "$@"
""")
            
            # Make terminal launcher executable
            os.chmod(terminal_launcher, 0o755)
            print(f"Created terminal launcher: {terminal_launcher}")

    print(f"Build complete! Executable is in {dist_dir}")
    
    # Final instructions
    print("\nImportant post-build steps:")
    if sys.platform == 'darwin':
        print(f"1. If you encounter issues with the app bundle, try running '{APP_NAME}_terminal.command' for debugging output")
        print(f"2. Ensure your .env file with API keys is properly placed in the app bundle")
    print(f"3. Make sure Neo4j and Java are properly bundled or will be downloaded on first run")
    
except Exception as e:
    print(f"Error during build process: {e}")
    import traceback
    traceback.print_exc()