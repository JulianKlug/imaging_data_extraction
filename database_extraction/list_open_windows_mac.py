import subprocess

def list_open_windows():
    script = '''
    set window_list to {}
    tell application "System Events"
        set process_list to (every process whose visible is true)
        repeat with proc in process_list
            set app_name to name of proc
            try
                set win_list to name of every window of proc
                repeat with win_name in win_list
                    copy (app_name & "||" & win_name) to end of window_list
                end repeat
            end try
        end repeat
    end tell
    return window_list
    '''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True
    )

    raw = result.stdout.strip()
    if not raw:
        return []

    # AppleScript returns comma-separated "App||Window" entries
    entries = [e.strip() for e in raw.split(", ") if e.strip()]
    windows = []
    for entry in entries:
        if "||" in entry:
            app, win = entry.split("||", 1)
            windows.append((app, win))
    return windows


if __name__ == "__main__":
    for app, window in list_open_windows():
        print(f"{app}: {window}")
