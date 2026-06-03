"""Meeting analysis task family — diagnostics plugin."""

from __future__ import annotations

from ..protocol import TaskPlugin, register_plugin


# task_id → expected output filename. Migrated from agent_loop/meeting_diagnostics.py.
EXPECTED_OUTPUT_FILE: dict[str, str] = {
    "task_meeting_advisory_acronyms": "acronym_glossary.md",
    "task_meeting_advisory_attendees": "attendees.md",
    "task_meeting_advisory_stakeholders": "stakeholder_analysis.md",
    "task_meeting_advisory_technical": "technical_discussions.md",
    "task_meeting_advisory_timeline": "timeline.md",
    "task_meeting_blog_post": "blog_post.md",
    "task_meeting_council_budget": "budget_analysis.md",
    "task_meeting_council_contact_info": "council_contacts.md",
    "task_meeting_council_neighborhood": "neighborhood_report.md",
    "task_meeting_council_public_comment": "public_comments_report.md",
    "task_meeting_council_upcoming": "upcoming_items.md",
    "task_meeting_council_votes": "votes_report.md",
    "task_meeting_executive_summary": "executive_summary.md",
    "task_meeting_follow_up_email": "follow_up_email.md",
    "task_meeting_gov_controversy": "controversy_analysis.md",
    "task_meeting_gov_data_sources": "data_sources.md",
    "task_meeting_gov_next_steps": "next_steps.md",
    "task_meeting_gov_qa_extract": "qa_exchanges.md",
    "task_meeting_gov_recommendations": "recommendations.md",
    "task_meeting_gov_speaker_summary": "speaker_summary.md",
    "task_meeting_searchable_index": "meeting_index.md",
    "task_meeting_sentiment_analysis": "sentiment_analysis.md",
    "task_meeting_tech_action_items": "action_items.md",
    "task_meeting_tech_competitors": "competitor_analysis.md",
    "task_meeting_tech_decisions": "decisions.md",
    "task_meeting_tech_messaging": "messaging_framework.md",
    "task_meeting_tech_product_features": "feature_priorities.md",
    "task_meeting_tldr": "meeting_tldr.md",
}

EXPECTED_INPUT_FILES: set[str] = {
    "transcript.md",
    "meeting_transcript.md",
    "meeting-transcript.md",
}


PLUGIN = TaskPlugin(
    family_id="meeting_analysis",
    expected_output_file=EXPECTED_OUTPUT_FILE,
    expected_input_files=EXPECTED_INPUT_FILES,
    task_id_prefix_match=("task_meeting_",),
)

register_plugin(PLUGIN)
