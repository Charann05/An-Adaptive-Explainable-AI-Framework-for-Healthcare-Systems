import subprocess
import sys
import threading
import time


def show_spinner(stop_event, message):
    symbols = ["|", "/", "-", "\\"]
    i = 0

    while not stop_event.is_set():
        print(f"\r{message} {symbols[i % len(symbols)]}", end="", flush=True)

        i += 1
        time.sleep(0.15)

    print("\r" + " " * (len(message) + 3), end="\r")


def install_package(package_name):
    print(f"\nInstalling {package_name}...")

    stop_event = threading.Event()
    spinner = threading.Thread(target=show_spinner,args=(stop_event, f"Installing {package_name}"))

    spinner.start()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    stop_event.set()
    spinner.join()

    if result.returncode == 0:
        print(f"✓ {package_name} installed successfully.")
        return True

    print(f"✗ Failed to install {package_name}.")
    print(result.stderr)

    return False