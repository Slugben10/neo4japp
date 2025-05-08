
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
