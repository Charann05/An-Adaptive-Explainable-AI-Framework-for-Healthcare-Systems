import os
from datetime import datetime

def generate_report(final_evidence, output_file="model_report.txt"):

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("MODEL ANALYSIS REPORT\n")
        file.write("=" * 50 + "\n")

        file.write(f"Generated At: {datetime.now()}\n\n")

        for category, evidence in final_evidence.items():
            if evidence is None:
                file.write(f"{category.capitalize()}: Unknown\n")
                continue

            file.write(
                f"{category.capitalize()}: {evidence['value']} ({evidence['confidence']}%)\n"
            )

    print(f"Report generated: {output_file}")
    print("Report saved at:", os.path.abspath(output_file))

    print("\n" + "=" * 50)

    with open(output_file, "r", encoding="utf-8") as file:
        print(file.read())