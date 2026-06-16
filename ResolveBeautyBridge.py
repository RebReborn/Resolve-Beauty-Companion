import sys
import os
import subprocess

def get_resolve_clip_path():
    # Dynamically set scripting paths if missing
    if not os.getenv("RESOLVE_SCRIPT_LIB"):
        os.environ["RESOLVE_SCRIPT_LIB"] = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
    if not os.getenv("RESOLVE_SCRIPT_API"):
        os.environ["RESOLVE_SCRIPT_API"] = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"

    # Attempt to import scripting module
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        # Check standard environment variables or Windows path
        api_path = os.getenv("RESOLVE_SCRIPT_API")
        if api_path:
            sys.path.append(api_path)
        else:
            sys.path.append(r"C:\Program Files\Blackmagic Design\DaVinci Resolve\developer\Scripting\Modules")
            
        try:
            import DaVinciResolveScript as dvr
        except ImportError:
            print("Error: DaVinciResolveScript module not found.")
            print("To use Resolve timeline integration, please ensure DaVinci Resolve is running and scripting is enabled in Preferences.")
            return None

    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        print("Error: Could not connect to Resolve application instance.")
        return None

    pm = resolve.GetProjectManager()
    proj = pm.GetCurrentProject()
    if not proj:
        print("Error: No active project found in Resolve.")
        return None

    timeline = proj.GetCurrentTimeline()
    if not timeline:
        print("Error: No active timeline found in the project.")
        return None

    # Get active clip under playhead
    clip = timeline.GetCurrentVideoItem()
    if not clip:
        print("Error: No clip selected under the timeline playhead.")
        return None

    media_item = clip.GetMediaPoolItem()
    if not media_item:
        print("Error: Timeline clip has no linked media pool item.")
        return None

    # Retrieve source file path
    path = media_item.GetClipProperty("File Path")
    if not path:
        path = media_item.GetClipProperty("Path")

    if not path or not os.path.exists(path):
        print(f"Error: Invalid or inaccessible media path: {path}")
        return None

    return path

def main():
    clip_path = get_resolve_clip_path()
    if not clip_path:
        sys.exit(1)

    print(f"Successfully bridged clip from Resolve: {clip_path}")
    
    # Locate application files
    app_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(app_dir, "main.py")
    
    # Try virtual environment Python executable first
    python_exe = os.path.join(app_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # fallback

    # Launch application in background passing clip path as argument
    try:
        subprocess.Popen([python_exe, main_py, clip_path])
        print("Resolve Beauty Companion launched successfully.")
    except Exception as e:
        print(f"Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
