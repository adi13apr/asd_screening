import json
from clinical_agent.retriever import retrieve_evidence
from clinical_agent.prompt_template import CLINICAL_PROMPT
from langchain_ollama import OllamaLLM


# Lightweight, CPU-friendly model
llm = OllamaLLM(model="phi3:mini")


def generate_clinical_explanation_from_file(
    fusion_output_path="outputs/fusion_output.json"
):
    # Load runtime fusion output
    with open(fusion_output_path, "r") as f:
        fusion_output = json.load(f)

    screening_summary = "\n".join(
        f"- {key}: {value}"
        for key, value in fusion_output["evidence_summary"].items()
    )

    query = (
        "autism EEG frontal temporal activity "
        "eye gaze visual attention joint attention "
        "gesture motor behavior early screening clinical studies"
    )

    docs = retrieve_evidence(query)

    evidence_blocks = []
    citations = set()

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")

        citations.add(f"{source} (page {page})")

        evidence_blocks.append(
            f"[Evidence {i+1}]\n"
            f"Source: {source}, page {page}\n"
            f"{doc.page_content[:500]}"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    prompt = CLINICAL_PROMPT.format(
        screening_summary=screening_summary,
        evidence=evidence_text
    )

    response = llm.invoke(prompt)

    return {
        "asd_risk_score": fusion_output["asd_risk_score"],
        "risk_level": fusion_output["risk_level"],
        "clinical_support_explanation": response,
        "sources_cited": sorted(list(citations)),
        "disclaimer": (
            "This output provides screening-level clinical decision support only "
            "and does not constitute a medical diagnosis."
        )
    }
