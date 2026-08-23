def print_banner():
    print(r"""
    ────────────────────────────────────────────────────────────────────────────────────────────────────
        ▄▖ ▌    ▗ ▘      ▖▖▄▖▄▖  ▄▖                ▌   ▐▘      ▖▖    ▜ ▗ ▌           ▄▖    ▗        
        ▌▌▛▌▀▌▛▌▜▘▌▌▌█▌  ▚▘▌▌▐   ▙▖▛▘▀▌▛▛▌█▌▌▌▌▛▌▛▘▙▘  ▜▘▛▌▛▘  ▙▌█▌▀▌▐ ▜▘▛▌▛▘▀▌▛▘█▌  ▚ ▌▌▛▘▜▘█▌▛▛▌▛▘
        ▛▌▙▌█▌▙▌▐▖▌▚▘▙▖  ▌▌▛▌▟▖  ▌ ▌ █▌▌▌▌▙▖▚▚▘▙▌▌ ▛▖  ▐ ▙▌▌   ▌▌▙▖█▌▐▖▐▖▌▌▙▖█▌▌ ▙▖  ▄▌▙▌▄▌▐▖▙▖▌▌▌▄▌
            ▌                                                                          ▄▌    
    ────────────────────────────────────────────────────────────────────────────────────────────────────       
    """)


def print_status(message):
    print(f"[✓] {message}")

def print_error(message):
    print(f"[✗] {message}")


def print_section(title):
    print("\n" + "─" * 62)
    print(f"{title:^62}")
    print("─" * 62)


def print_step(message):
    print(f"  → {message}")


# def print_startup():
#     print_status("Initializing framework...")
#     print_status("Source project detected")
#     print_status("Loading analyzers...")
#     print_status("Loading XAI modules...")


# def print_analysis_steps():
#     print_section("ANALYSIS STARTED")

#     print_step("Scanning project files")
#     print_step("Detecting model framework")
#     print_step("Detecting architecture")
#     print_step("Detecting input type")
#     print_step("Detecting features")
#     print_step("Building evidence")
#     print_step("Generating model report")

#     print("\n" + "─" * 62)