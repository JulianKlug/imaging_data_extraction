from __future__ import annotations
from pathlib import Path
import time
import os
import sys
import pyautogui

from list_open_windows_mac import list_open_windows
from launch_pacs import launch_osirix_pacs, kill_osirix_pacs
from imaging_extraction import ensure_osirix_open

def wait_until_empty_for(dir_path: str | Path, T: float, poll: float = 0.5,
                         coordinates=None, path_to_identifiers_file=None, verbose=False) -> bool:
    """
    Return True once `dir_path` has stayed empty continuously for >= T seconds.
    (Resets the empty-timer if anything appears.)
    """
    d = Path(dir_path)
    if not d.is_dir():
        raise NotADirectoryError(d)

    empty_since = None
    while True:
        is_empty = not any(d.iterdir())
        now = time.monotonic()

        if is_empty:
            if empty_since is None:
                empty_since = now
            if now - empty_since >= T:
                return True
        else:
            empty_since = None

        windows = list_open_windows()
        osirix_windows = [w for w in windows if w[0] == "OsiriX MD"]
        if len(osirix_windows) > 2:
            pyautogui.click(coordinates['osirix_popup_ok'])
        ensure_osirix_open(path_to_identifiers_file, verbose=verbose)

        time.sleep(poll)


def ensure_osirix_open(path_to_identifiers_file,
                      verbose=False):
    windows = list_open_windows()
    osirix_windows = [w for w in windows if w[0] == "OsiriX MD"]    
    if len(osirix_windows) < 2:
        if path_to_identifiers_file is None:
            raise Exception('Osirix or Compacs are not open!')
        else:
            print('Osirix or Compacs are not open! Attempting to re-open Osirix...')
            kill_osirix_pacs()
            launch_osirix_pacs(path_to_identifiers_file, verbose=verbose)
    return 

def verify_modal_failsafe(safe_mode=True):
    if not safe_mode:
        return
    if verify_modal_presence():
        sys.exit("Error message")


def verify_modal_presence():
    """
    Verify if a modal dialog is present on the screen.
    """
    # modals to verify 
    # check all png files in modal directory (current file is in the same directory as the modal directory)
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    modal_directory = os.path.join(current_file_directory, 'modals')
    modal_files = [f for f in os.listdir(modal_directory) if f.endswith('.png')]
    for modal_file in modal_files:
        modal_path = os.path.join(modal_directory, modal_file)
        try:
            if pyautogui.locateOnScreen(modal_path) is not None:
                print(f'Modal dialog {modal_file} is present on the screen.')
                return True
        except Exception as e:
            pass  # If the image is not found, continue to the next one

    return False
