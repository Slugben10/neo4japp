
# Hook to help PyInstaller find resources in macOS app bundle
import os
import sys

# Set environment variables to help the app find its resources
if getattr(sys, 'frozen', False):
    # We're running in a bundle
    bundle_dir = os.path.dirname(sys.executable)
    if '.app/Contents/MacOS' in bundle_dir:
        # We're in a macOS .app bundle
        resources_dir = os.path.abspath(os.path.join(bundle_dir, '..', 'Resources'))
        app_dir = os.path.abspath(os.path.join(bundle_dir, '..', '..', '..'))
        
        # Set environment variables
        os.environ['RA_RESOURCES_PATH'] = resources_dir
        os.environ['RA_APP_PATH'] = app_dir
