from clinical_agent import generate_clinical_explanation_from_file


if __name__ == "__main__":
    print("\n=== Running Clinical Support Agent ===\n")

    result = generate_clinical_explanation_from_file(
        fusion_output_path="outputs/fusion_output.json"
    )

    print("=== CLINICAL SUPPORT OUTPUT ===\n")
    print(result)
