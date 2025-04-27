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
        importlib.import_module(package)
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
        "system_prompt": "You are a helpful AI research assistant. Your goal is to help researchers write new papers or expand work-in-progress papers based on the provided documents and instructions."
    }
    
    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=2)
    print("Default config.json created successfully")

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

# Create general app hook
app_hook_path = os.path.join("hooks", "hook-app.py")
with open(app_hook_path, 'w') as f:
    f.write("""
# General application hook
import os
import sys

# Ensure we can find the app's resources
if getattr(sys, 'frozen', False):
    # Running as a bundled executable
    APP_PATH = os.path.dirname(sys.executable)
    os.environ['RA_APP_PATH'] = APP_PATH
""")

# Create macOS specific hook for path detection
macos_hook_path = os.path.join("hooks", "hook-macos-paths.py")
with open(macos_hook_path, 'w') as f:
    f.write("""
# macOS specific path detection hook
import os
import sys

# Add different possible paths for resources when in a macOS bundle
if getattr(sys, 'frozen', False) and sys.platform == 'darwin':
    # We're in a macOS app bundle
    bundle_path = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    resources_path = os.path.join(bundle_path, 'Resources')
    macos_path = os.path.join(bundle_path, 'MacOS')
    
    # Add these paths to sys.path to help with imports
    sys.path.insert(0, bundle_path)
    sys.path.insert(0, resources_path)
    sys.path.insert(0, macos_path)
    
    # Set environment variables to help find resources
    os.environ['RA_BUNDLE_PATH'] = bundle_path
    os.environ['RA_RESOURCES_PATH'] = resources_path
    os.environ['RA_MACOS_PATH'] = macos_path
""")

# Define data files to include
data_files = [
    ("config.json", "."),
    ("install_java.py", "."),  # Include Java installer script
    ("force_java_config.py", "."),  # Include Java force config script
]

# Add .env file if it exists
if os.path.exists(".env"):
    data_files.append((".env", "."))

# Ensure the jre directory exists in the output folder
dist_dir = os.path.join('dist', APP_NAME)
jre_dir = os.path.join(dist_dir, 'jre')
os.makedirs(jre_dir, exist_ok=True)

# Ensure that the install_java.py script is executable
if os.path.exists("install_java.py"):
    try:
        if sys.platform != 'win32':  # Unix-like systems
            os.chmod("install_java.py", 0o755)
    except Exception as e:
        print(f"Warning: Could not make install_java.py executable: {e}")

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

# Additional wxPython-specific imports
wxpy_modules = [
    "wx.adv",
    "wx.html",
    "wx.grid",
    "wx.xrc",
    "wx._xml",
    "wx._html",
    "wx._adv",
    "wx._core",
    "wx._controls"
]
hidden_imports.extend(wxpy_modules)

# Base PyInstaller arguments
pyinstaller_args = [
    'main.py',
    '--name=' + APP_NAME,
    '--onedir',
    '--clean',
    '--noconfirm',
    '--noconsole',  # Elrejtjük a konzolablakot
]

# Add runtime hooks
pyinstaller_args.append('--runtime-hook=hooks/hook-app.py')
pyinstaller_args.append('--runtime-hook=hooks/hook-macos-paths.py')

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
    
    # Exclude modules that might cause conflicts
    pyinstaller_args.append('--exclude-module=tkinter')
    pyinstaller_args.append('--exclude-module=PySide')
    pyinstaller_args.append('--exclude-module=PyQt5')
    
    # Add icon if available
    if os.path.exists('app_icon.icns'):
        pyinstaller_args.append('--icon=app_icon.icns')
elif sys.platform == 'win32':  # Windows
    print("Building for Windows...")
    
    # Add specific Windows options for wxPython
    pyinstaller_args.append('--hidden-import=wx.msw')
    
    # Windows-specifikus opciók az antivírus-jelzések elkerülésére
    pyinstaller_args.append('--uac-admin')
    
    # Add icon if available
    if os.path.exists('app_icon.ico'):
        pyinstaller_args.append('--icon=app_icon.ico')

# Print the PyInstaller command for debugging
print("PyInstaller command:", " ".join(pyinstaller_args))

try:
    # Run PyInstaller
    pyinstaller_main.run(pyinstaller_args)

    # Ensure the Documents directory exists in the output folder
    dist_dir = os.path.join('dist', APP_NAME)
    documents_dir = os.path.join(dist_dir, 'Documents')
    os.makedirs(documents_dir, exist_ok=True)
    
    # Ensure the Neo4jDB directory exists in the output folder
    neo4jdb_dir = os.path.join(dist_dir, 'Neo4jDB')
    os.makedirs(neo4jdb_dir, exist_ok=True)
    
    # Ensure the Prompts directory exists in the output folder
    prompts_dir = os.path.join(dist_dir, 'Prompts')
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Ensure the jre directory exists in the output folder
    jre_dir = os.path.join(dist_dir, 'jre')
    os.makedirs(jre_dir, exist_ok=True)
    
    # Create a README file in the jre directory
    jre_readme_path = os.path.join(jre_dir, 'README.txt')
    with open(jre_readme_path, 'w') as f:
        f.write("This directory will contain a Java Runtime Environment (JRE) for Neo4j.\n")
        f.write("If you experience issues with Neo4j, the application will help you install Java here.\n")
        f.write("You can also run install_java.py manually to install Java.\n")

    # Explicitly copy config.json to the output folder to ensure it's available
    if os.path.exists('config.json'):
        shutil.copy('config.json', dist_dir)
        print(f"Copied config.json to {dist_dir}")

    # Copy .env file to dist folder (if it exists)
    if os.path.exists('.env'):
        shutil.copy('.env', dist_dir)
        print(f"Copied .env to {dist_dir}")

    # For macOS, perform additional compatibility fixes
    if sys.platform == 'darwin':
        app_bundle_path = os.path.join('dist', f"{APP_NAME}.app")
        if os.path.exists(app_bundle_path):
            print(f"Performing additional macOS compatibility fixes for {app_bundle_path}...")
            
            # Ensure config.json and .env are copied into the app bundle at multiple locations
            contents_path = os.path.join(app_bundle_path, "Contents")
            macos_path = os.path.join(contents_path, "MacOS")
            resources_path = os.path.join(contents_path, "Resources")
            
            # Create directories if they don't exist
            os.makedirs(macos_path, exist_ok=True)
            os.makedirs(resources_path, exist_ok=True)
            
            # Create Documents directory in multiple locations
            for dest_path in [macos_path, resources_path]:
                doc_path = os.path.join(dest_path, "Documents")
                os.makedirs(doc_path, exist_ok=True)
                print(f"Created Documents directory at: {doc_path}")
                
                # Create Neo4jDB directory
                neo4jdb_path = os.path.join(dest_path, "Neo4jDB")
                os.makedirs(neo4jdb_path, exist_ok=True)
                print(f"Created Neo4jDB directory at: {neo4jdb_path}")
                
                # Create Prompts directory
                prompts_path = os.path.join(dest_path, "Prompts")
                os.makedirs(prompts_path, exist_ok=True)
                print(f"Created Prompts directory at: {prompts_path}")
                
                # Create JRE directory
                jre_path = os.path.join(dest_path, "jre")
                os.makedirs(jre_path, exist_ok=True)
                print(f"Created JRE directory at: {jre_path}")
                
                # Create README in JRE directory
                jre_readme_path = os.path.join(jre_path, 'README.txt')
                with open(jre_readme_path, 'w') as f:
                    f.write("This directory will contain a Java Runtime Environment (JRE) for Neo4j.\n")
                    f.write("If you experience issues with Neo4j, the application will help you install Java here.\n")
            
            # Copy config.json to multiple locations
            for dest_path in [macos_path, resources_path]:
                if os.path.exists('config.json'):
                    shutil.copy('config.json', dest_path)
                    print(f"Copied config.json to {dest_path}")
            
            # Copy install_java.py to ensure it's available
            if os.path.exists('install_java.py'):
                shutil.copy('install_java.py', dest_path)
                # Make it executable
                os.chmod(os.path.join(dest_path, 'install_java.py'), 0o755)
                print(f"Copied install_java.py to {dest_path}")
            
            # Copy force_java_config.py to ensure it's available
            if os.path.exists('force_java_config.py'):
                shutil.copy('force_java_config.py', dest_path)
                # Make it executable
                os.chmod(os.path.join(dest_path, 'force_java_config.py'), 0o755)
                print(f"Copied force_java_config.py to {dest_path}")
            
            # Copy .env to multiple locations
            if os.path.exists('.env'):
                for dest_path in [macos_path, resources_path]:
                    shutil.copy('.env', dest_path)
                    print(f"Copied .env to {dest_path}")
            
            # Create a fixup script to run after installation
            fixup_script = os.path.join(dist_dir, "fix_macos_app.sh")
            with open(fixup_script, 'w') as f:
                f.write("""#!/bin/bash
# Fix for wxPython symbol issues on macOS
# This script should be run after installation if you encounter issues

APP_PATH="$(cd "$(dirname "$0")" && pwd)"
echo "Fixing wxPython compatibility issues in: $APP_PATH"

# Fix library paths if needed
install_name_tool -change @loader_path/libwx_baseu-3.1.dylib @executable_path/libwx_baseu-3.1.dylib "$APP_PATH/wx/_core.so" 2>/dev/null || true
install_name_tool -change @loader_path/libwx_osx_cocoau-3.1.dylib @executable_path/libwx_osx_cocoau-3.1.dylib "$APP_PATH/wx/_core.so" 2>/dev/null || true

# Create symbolic links to ensure files can be found
BUNDLE_DIR="$(dirname "$(dirname "$APP_PATH")")"
echo "Creating symbolic links in bundle directory: $BUNDLE_DIR"

# Link .env and config.json if they exist
[ -f "$APP_PATH/.env" ] && ln -sf "$APP_PATH/.env" "$BUNDLE_DIR/.env" 2>/dev/null || true
[ -f "$APP_PATH/config.json" ] && ln -sf "$APP_PATH/config.json" "$BUNDLE_DIR/config.json" 2>/dev/null || true

echo "Fix completed. Try running the app again."
""")
            
            # Make the script executable
            os.chmod(fixup_script, 0o755)
            print(f"Created macOS compatibility fix script at {fixup_script}")

    print(f"Build complete. Executable is in {dist_dir}")
    print(f"Documents directory created at {documents_dir}")
    
except Exception as e:
    print(f"Error during build process: {e}")
    import traceback
    traceback.print_exc()

print("\nIf you encounter issues with the app:")
print("1. Make sure wxPython is properly installed: pip install -U wxPython")
print("2. Make sure PyInstaller and its dependencies are installed: pip install -U pyinstaller altgraph")
print("3. Try running the app from terminal to see any error output")

if sys.platform == 'darwin':
    print("\nFor macOS-specific wxPython issues:")
    print("1. Try running the fix_macos_app.sh script in the application directory")
    print("2. Ensure you're using a compatible version of wxPython for your macOS version")
    print("3. Consider using PyInstaller 4.10 or newer for better macOS compatibility")
    print("4. If using the app bundle, try running it from the terminal with:")
    print(f"   open -a {APP_NAME}")