import json
import subprocess
from clinical_agent.clinical_agent import generate_clinical_explanation_from_file


def run_screening_agent():
    print("\n=== Running Screening Agent ===\n")
    subprocess.run(["python", "full_pipeline.py"], check=True)


def run_clinical_support_agent():
    print("\n=== Running Clinical Support Agent ===\n")
    return generate_clinical_explanation_from_file(
        fusion_output_path="outputs/fusion_output.json"
    )


def integrate_agents():
    # 1. Run screening
    run_screening_agent()

    # 2. Load screening output
    with open("outputs/fusion_output.json", "r") as f:
        screening_output = json.load(f)

    # 3. Run clinical agent
    clinical_output = run_clinical_support_agent()

    # 4. Combine outputs clearly
    final_output = {
        "screening_agent_output": screening_output,
        "clinical_support_agent_output": clinical_output
    }

    # 5. Save final result
    with open("outputs/final_decision_support.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n=== FINAL DECISION SUPPORT OUTPUT ===\n")
    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    integrate_agents()
