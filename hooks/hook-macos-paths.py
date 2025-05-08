
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
