from .evidence import create_evidence_store
from .load_model_files import load_files
from .classify_file_type import get_file_info
from .read_config import scan_config
from .read_requirements import scan_requirements
from .read_source import scan_file
from .read_input_type import detect_input_type
from .evidence_reducer import reduce_evidence
from .evidence_ranker import rank_evidence
from .report_generator import generate_report


def analyze_model():
    evidences = create_evidence_store()

    MODEL_FOLDER, files = load_files()
    file_info = get_file_info(MODEL_FOLDER, files)

    scan_config(MODEL_FOLDER, evidences, file_info)
    scan_requirements(MODEL_FOLDER, evidences, file_info)
    scan_file(MODEL_FOLDER, evidences, file_info)
    detect_input_type(MODEL_FOLDER, evidences, file_info)

    print("Evidences found:\n", evidences)

    evidences = reduce_evidence(evidences)
    print("Reduced Evidences:", evidences)

    final_evidence = rank_evidence(evidences)

    print("\n" + "=" * 50)
    print("FINAL EVIDENCE")
    print("=" * 50)

    for category, evidence in final_evidence.items():
        label = category.replace("_", " ").title()

        if evidence is None:
            print(f"{label:<15}: Unknown")
        else:
            print(f"{label:<15}: {evidence['value']}")

    print("=" * 50)
    generate_report(final_evidence)

    return final_evidence


if __name__ == "__main__":
    analyze_model()