import subprocess
import os
from tabnanny import verbose
import time
import pyautogui

OSIRIX_LAUNCH_WAIT_TIME = 10  # seconds

COORDINATES = {
    "reference": (14, 40), # reference of close window cross 
    'version_accept': (1104, 526),
    'window': (319, 39),
    'compacs': (35, 82),
}


def launch_osirix_pacs(path_to_identifiers_file, verbose=False):
    # Define the path to the OsiriX PACS application
    osirix_pacs_path = "/Applications/OsiriX MD.app"

    # Check if the OsiriX PACS application exists at the specified path
    if not os.path.exists(osirix_pacs_path):
        raise FileNotFoundError(f"OsiriX PACS application not found at {osirix_pacs_path}")
    
    # ensure the identifiers file exists
    if not os.path.exists(path_to_identifiers_file):
        raise FileNotFoundError(f"Identifiers file not found at {path_to_identifiers_file}")
    
    # load id and pwd from identifiers file
    username = None
    password = None
    with open(path_to_identifiers_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError("Identifiers file must contain at least two lines: username and password")
        username = lines[0].strip()
        password = lines[1].strip()

    # Launch the OsiriX PACS application
    subprocess.run(["open", "-n", osirix_pacs_path])

    time.sleep(OSIRIX_LAUNCH_WAIT_TIME/3)
    pyautogui.write(password, interval=0.1)
    pyautogui.press("enter")

    # wait for a short period to ensure the application has time to launch
    time.sleep(OSIRIX_LAUNCH_WAIT_TIME)

    # click on accept osirix version
    if verbose:
        print("Clicking on accept osirix version")
    pyautogui.click(COORDINATES['version_accept'])

    # click on osirix window to make it active
    if verbose:
        print("Clicking on osirix window to make it active")
    pyautogui.click(COORDINATES['window'])

    # click on compacs
    if verbose:
        print("Clicking on compacs")
    pyautogui.click(COORDINATES['compacs'])

    # enter username and password
    if verbose:
        print("Entering username and password")
    pyautogui.write(username, interval=0.1)
    pyautogui.press("tab")
    pyautogui.write(password, interval=0.1)
    pyautogui.press("enter")

    # select role (thrice downward arrow and enter)
    if verbose:
        print("Selecting role")
    pyautogui.press("down")
    pyautogui.press("down")
    pyautogui.press("down")
    pyautogui.press("enter")
    pyautogui.press("enter")


def kill_osirix_pacs():
    subprocess.run(["pkill", "-9", "OsiriX MD"])