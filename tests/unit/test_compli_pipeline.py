from compli_pipeline import CompliBotPipeline


def make_pipeline() -> CompliBotPipeline:
    """
    Create the pipeline without initializing ChromaDB or loading the
    SentenceTransformer model. These tests exercise deterministic logic only.
    """
    return object.__new__(CompliBotPipeline)


def test_classify_question_detects_compliance_question_types():
    pipeline = make_pipeline()

    assert pipeline.classify_question("What is CAPA?") == "Definition"
    assert pipeline.classify_question("How does the review process work?") == "Procedure"
    assert (
        pipeline.classify_question("What training is required?")
        == "Training"
    )
    assert (
        pipeline.classify_question("How should a deviation be escalated?")
        == "Escalation / Quality Event"
    )


def test_detect_doc_group_uses_expected_filename_categories():
    pipeline = make_pipeline()

    assert pipeline.detect_doc_group("SOP_Deviation_Handling.pdf") == "sop"
    assert pipeline.detect_doc_group("ICH_Guideline.pdf") == "guideline"
    assert pipeline.detect_doc_group("Quality_Manual.pdf") == "quality_doc"
    assert pipeline.detect_doc_group("misc_document.pdf") == "general"


def test_infer_question_doc_preference():
    pipeline = make_pipeline()

    assert (
        pipeline.infer_question_doc_preference(
            "What does this SOP say about deviation handling?"
        )
        == "sop"
    )

    assert (
        pipeline.infer_question_doc_preference(
            "What does the FDA guideline require?"
        )
        == "guideline"
    )

    assert pipeline.infer_question_doc_preference("What is documented here?") == "any"


def test_this_does_not_trigger_casual_chat_false_positive():
    pipeline = make_pipeline()

    chunks = [
        {
            "text": (
                "This SOP describes the deviation process and requires "
                "documented investigation and quality review."
            ),
            "source": "SOP_Deviation.pdf",
            "chunk_index": 0,
            "doc_group": "sop",
            "distance": 0.8,
        }
    ]

    result = pipeline.evaluate_grounding(
        "What does this SOP say about deviation handling?",
        chunks,
    )

    assert result["status"] != "not_grounded"
    assert result["reason"] != "casual_chat"


def test_actual_casual_chat_is_not_grounded():
    pipeline = make_pipeline()

    chunks = [
        {
            "text": (
                "This SOP describes deviation investigation, review, "
                "approval, documentation, and closure requirements."
            ),
            "source": "SOP_Deviation.pdf",
            "chunk_index": 0,
            "doc_group": "sop",
            "distance": 0.8,
        }
    ]

    result = pipeline.evaluate_grounding("Hi, how are you?", chunks)

    assert result == {
        "status": "not_grounded",
        "reason": "casual_chat",
    }


def test_chunking_retains_section_content():
    pipeline = make_pipeline()

    text = (
        "Purpose: This procedure defines deviation handling responsibilities "
        "for regulated operations and quality personnel. "
        "Procedure: Deviations must be documented, investigated, reviewed, "
        "approved, and closed according to the applicable quality process."
    )

    chunks = pipeline.chunk_text(
        text,
        chunk_size=120,
        overlap_sentences=1,
    )

    assert chunks
    combined = " ".join(chunks).lower()

    assert "deviation" in combined
    assert "documented" in combined
    assert "reviewed" in combined


def test_synthesize_answer_returns_structured_document_grounded_output():
    pipeline = make_pipeline()

    retrieved = [
        {
            "text": (
                "Deviations must be documented and investigated. "
                "Quality review is required before closure."
            ),
            "source": "SOP_Deviation.pdf",
            "chunk_index": 2,
            "doc_group": "sop",
            "distance": 0.75,
        },
        {
            "text": (
                "The investigator shall document the outcome and obtain "
                "appropriate approval before closure."
            ),
            "source": "SOP_Deviation.pdf",
            "chunk_index": 3,
            "doc_group": "sop",
            "distance": 0.82,
        },
    ]

    result = pipeline.synthesize_answer(
        "What does this SOP require for deviation closure?",
        retrieved,
    )

    assert result["answer_summary"]
    assert result["procedure_guidance"]
    assert result["key_requirements"]
    assert result["evidence"]
    assert result["source"].startswith("SOP_Deviation.pdf")
    assert "review" in result["compliance_note"].lower()
