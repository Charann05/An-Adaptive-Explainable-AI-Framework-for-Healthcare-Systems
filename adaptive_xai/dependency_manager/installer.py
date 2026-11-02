import subprocess
import sys
import threading
import time

INSTALL_TIMEOUT_SECONDS = 120


def show_spinner(stop_event, message):
    symbols = ["|", "/", "-", "\\"]
    i = 0

    while not stop_event.is_set():
        print(f"\r{message} {symbols[i % len(symbols)]}", end="", flush=True)

        i += 1
        time.sleep(0.15)

    print("\r" + " " * (len(message) + 3), end="\r")


def install_package(package_name, timeout=INSTALL_TIMEOUT_SECONDS):
    stop_event = threading.Event()
    spinner = threading.Thread(
        target=show_spinner,
        args=(stop_event, f"Installing {package_name}"),
    )
    spinner.start()

    result = None
    failure = None
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        failure = (
            f"Installation timed out after {timeout} seconds. "
            "Check your network connection and try again."
        )
        if error.stderr:
            stderr = error.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            failure = f"{failure}\n{stderr.strip()}"
    except OSError as error:
        failure = f"Could not start pip: {error}"
    finally:
        stop_event.set()
        spinner.join()

    if result is not None and result.returncode == 0:
        print(f"✓ {package_name} installed successfully.")
        return True

    print(f"✗ Failed to install {package_name}.")
    if failure:
        print(failure)
    elif result is not None and result.stderr:
        print(
            "pip installation failed. Check your network connection, "
            "package name, and pip output."
        )
        print(result.stderr.strip())

    return False