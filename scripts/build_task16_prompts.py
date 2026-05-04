#!/usr/bin/env python3
"""
Build a task_16 RL prompt pool from the canonical task plus mixed prompt variants.

Design goals:
- Keep the canonical task_id so workspace/grading stay unchanged.
- Reuse the old/base prompt pool rather than replacing it.
- Add targeted prompt groups derived from the failure taxonomy.
- Vary only the user-facing prompt wording and emphasis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib_tasks import resolve_task_markdown_path  # noqa: E402
from rl.train.prepare_prompts import build_verl_row, parse_task_file  # noqa: E402


BASE_PROMPTS = [
    # canonical-like, but avoid over-rewarding early closure
    "Triage the emails in this inbox and write `triage_report.md` with priority, category, rationale, and recommended action for each email.",
    "Process the inbox, preserve evidence from each email, group related issues when appropriate, and produce `triage_report.md`.",
    # evidence preservation
    "Before summarizing, keep an audit trail in the final report: every email id must appear with the concrete clue that drove its priority.",
    "Do not replace the inbox with generic triage templates. Use the actual sender, subject, deadline, customer, or incident evidence from each email.",
    # complete coverage
    "Cover the entire inbox in `triage_report.md`. Every email should be accounted for, even if some are low priority.",
    "Write `triage_report.md` so that no inbox item is left untriaged. Include low-priority items briefly, but do not omit them.",
    # incident linkage
    "Triage this inbox with special attention to related operational incidents. If multiple emails refer to the same outage/alert, link them in `triage_report.md`.",
    "As you triage the inbox, connect emails that belong to the same incident and capture that linkage explicitly in `triage_report.md`.",
    # business and security weighting
    "Prioritize with business judgment. Important customer, outage, security, and release-blocking items should clearly outrank routine internal or newsletter mail.",
    "When triaging, weight production outages, security deadlines, high-value customer threads, and release blockers appropriately.",
    # low-value separation
    "Separate low-value noise from real work. Spam, newsletters, and routine notifications should not be elevated above incidents, customers, or security deadlines.",
    "Keep low-priority emails visible but brief, and make sure they do not crowd out urgent operational or customer-facing work.",
    # report quality
    "The final report should be evidence-grounded: each priority and action should be traceable to a specific email detail.",
    "Produce a concise but complete operator handoff that preserves both complete per-email coverage and the key incident/customer/security context.",
]

TARGETED_PROMPT_GROUPS: dict[str, list[str]] = {
    "email13_coverage": [
        "Do not omit low-salience but operationally important alerts. Make sure every inbox item, including correlated monitoring alerts, appears in `triage_report.md`.",
        "When writing `triage_report.md`, account for every email explicitly. A monitoring alert that relates to an active outage must not be skipped.",
        "Cover the whole inbox. If a message looks like a supporting alert for a production issue, include it explicitly instead of treating it as disposable noise.",
        "Make sure `triage_report.md` includes all emails, including alerts that may look minor on their own but matter in operational context.",
    ],
    "incident_linkage": [
        "If an outage email and a monitoring alert refer to the same operational problem, connect them explicitly in `triage_report.md` rather than listing them as unrelated items.",
        "Related outage and alert emails should be grouped into one incident view in `triage_report.md` whenever the evidence supports it.",
        "While triaging, look for emails that belong to the same incident and make that linkage explicit in the report.",
        "Do not treat correlated outage and alert threads as isolated events. Show incident linkage clearly in `triage_report.md`.",
    ],
    "email13_priority": [
        "A correlated latency alert during an ongoing outage should rank near the top, not with routine low-priority mail. Reflect that in `triage_report.md`.",
        "Do not downgrade a monitoring alert that is tied to an active production incident. Prioritize it appropriately in the final report.",
        "When an alert appears to confirm or extend an outage, rank it as an operationally important item rather than background noise.",
        "Use operational context when assigning priority: a correlated alert during an incident should not be buried in the queue.",
    ],
    "bigclient_weighting": [
        "High-value customer threads should stand out clearly in `triage_report.md`. Customer impact and revenue risk must affect priority.",
        "Do not treat a major customer thread like routine mail. Surface high-value customer impact explicitly in the final triage.",
        "When business impact is high, rank customer-facing issues above routine internal requests and administrative threads.",
        "Use revenue and customer-risk weighting in your prioritization. Important customer emails must be clearly surfaced in `triage_report.md`.",
    ],
    "security_weighting": [
        "Security and compliance deadlines require elevated priority and a concrete next action in `triage_report.md`.",
        "Do not file security/compliance work as ordinary admin mail. Give it explicit priority and follow-up action.",
        "When a message carries security or compliance urgency, reflect that with a higher rank and a specific recommended action.",
        "Security-sensitive or compliance-related items should be clearly elevated above routine inbox noise in the final report.",
    ],
    "evidence_preservation": [
        "Do not write a generic triage template. The report must preserve concrete evidence from the actual emails, such as sender, subject, deadline, customer value, outage signal, or release blocker.",
        "A good triage report should be auditable. For each email or group, include the email id and the factual clue that justifies the priority.",
        "When compressing the inbox into a report, do not lose the evidence. Keep enough detail that another operator can see why each priority was chosen.",
        "Avoid placeholder entries. Each report item should name the real email id and a real clue from that email.",
    ],
    "coverage_before_summary": [
        "Do not let the summary replace the report. First ensure all 13 email ids are represented, then add a short summary.",
        "The final artifact must not be summary-only. Include each email id with priority, category, and action before or alongside any high-level summary.",
        "A report that only lists the top few emails is incomplete. Cover all 13 emails, including routine or low-value items.",
        "Preserve complete coverage while staying concise: one specific line per low-priority email is enough, but omission is not acceptable.",
    ],
    "incident_graph": [
        "Before assigning final priorities, first build an incident graph for the inbox: determine which emails belong to the same operational event, which are standalone work items, and which are low-value noise. Then write `triage_report.md` from that global view.",
        "Do not triage each email in isolation. First identify incident groups across the inbox, then assign priorities from the incident level down to the email level in `triage_report.md`.",
        "Use a two-stage workflow: (1) build an inbox-level event map, grouping related outage/alert emails together, and (2) write `triage_report.md` from that event map rather than from isolated local judgments.",
        "Model the inbox as incidents plus standalone tasks. Group related emails first, then write the final report using that global grouping.",
    ],
    "priority_propagation": [
        "After identifying related emails, propagate priority from the incident level to all member emails. If one email establishes a P0 outage, related alert emails should not be ranked as routine low-priority items.",
        "Use incident-level priority propagation: when two emails belong to the same critical incident, their priorities must stay consistent with that shared incident severity.",
        "Do not let a correlated alert inherit a lower priority just because its wording looks less urgent. Incident membership should affect priority assignment in `triage_report.md`.",
        "Assign priority globally, not locally: critical incident context should raise the ranking of supporting alert emails and related follow-up threads.",
    ],
    "report_schema_incident_groups": [
        "Write `triage_report.md` with two top-level sections: `## Incident Groups` and `## Standalone Items`. Under `Incident Groups`, list each incident with member emails, shared priority, rationale, and action. Under `Standalone Items`, cover the remaining emails.",
        "Structure the report explicitly around grouped incidents. Use one section for grouped operational incidents and another for standalone items, so the global inbox state is visible.",
        "The final report must expose the inbox-level structure. Use `## Incident Groups` for linked outage/alert threads and `## Standalone Items` for everything else.",
        "Do not output only a flat per-email list. Make the report show incident grouping first, then standalone tasks, so priority decisions are traceable.",
    ],
    "report_schema_priority_fields": [
        "In `triage_report.md`, every incident group or standalone item must include: covered emails, priority, category, rationale, and recommended action.",
        "Use an operator-readable schema in the report: each grouped incident or standalone item should show member emails, priority, category, why it matters, and what to do next.",
        "Make the report schema explicit. For each incident group or standalone item, include the emails covered, the shared priority, the category, the reasoning, and the next action.",
        "The report should make global judgment legible: show which emails belong together and give each group or standalone item a clear priority, rationale, and action.",
    ],
    "anti_template": [
        "Do not invent generic examples such as customer order confirmations or placeholder incidents. The report must be based only on the actual inbox files.",
        "If you cannot cite an email id and a concrete clue, do not include that claim in the final report.",
        "Avoid hallucinated subjects, customers, or actions. Every item in `triage_report.md` should correspond to a real inbox email.",
        "The report should not contain bracket placeholders or generic sample rows; it should contain real triage decisions from the inbox.",
    ],
    "priority_audit": [
        "Before finalizing priorities, audit the top and bottom of the queue: production incidents and related alerts should stay high, while spam and newsletters should stay low.",
        "Check that high-value customer work, security deadlines, and release blockers have not been buried below routine administrative or automated mail.",
        "Priority should reflect evidence, not email position. Late inbox items can still be urgent, and early routine items can still be low priority.",
        "After assigning priorities, verify that P0/P1 items are truly urgent and that P4 items are genuinely safe to archive or defer.",
    ],
    "coverage_audit": [
        "Before writing the final answer, audit email_01 through email_13 and make sure each id appears exactly as a covered item or a member of an incident group.",
        "Use the email ids as a checklist. The final report should not lose any id while compressing related messages into incident groups.",
        "If an email is low priority, still include it with a short line; if it is grouped into an incident, list it under that incident group.",
        "The final report should make it easy to verify coverage: every inbox file should be traceable to a report entry.",
    ],
}


HELDOUT_VAL_PROMPT_GROUPS: dict[str, list[str]] = {
    "heldout_base": [
        "Perform a complete operator-style triage of the inbox and persist the result in `triage_report.md`.",
        "Inspect the inbox, decide what matters operationally, and leave a finished `triage_report.md` for the on-call owner.",
    ],
    "heldout_coverage": [
        "The report must account for all 13 inbox files. Briefly include low-value mail, but do not skip any exact email id.",
        "Treat coverage as part of the deliverable: every email_01 through email_13 should be represented in `triage_report.md`.",
    ],
    "heldout_incident_linkage": [
        "Use outage notes and monitoring alerts together as evidence for shared incidents, then document the linked emails in the report.",
        "If two messages point to the same customer-facing failure, report them as one incident instead of independent tasks.",
    ],
    "heldout_priority_propagation": [
        "When an alert corroborates a live production incident, give it the incident priority instead of ranking it as background noise.",
        "Assign priorities from the global incident picture: supporting alerts for a P0 event should stay visibly urgent.",
    ],
    "heldout_customer_security": [
        "Customer-facing revenue risk and same-day security work should outrank routine finance, HR, newsletter, or social mail.",
        "Elevate major customer blockers and security/compliance deadlines with concrete next actions in `triage_report.md`.",
    ],
    "heldout_schema": [
        "Organize the final artifact around grouped incidents first and standalone items second, with priority and action for each.",
        "Make the report easy to audit: show covered emails, category, priority, rationale, and recommended action for every group or item.",
    ],
    "heldout_workflow": [
        "After reading the inbox, write a complete evidence-grounded triage report rather than a generic summary.",
        "Do not end with analysis in chat. The task is complete only after `triage_report.md` exists, covers the inbox, and preserves key evidence.",
    ],
    "heldout_global_judgment": [
        "Build an inbox-level event map before writing final priorities so related operational evidence is not buried.",
        "Separate urgent incident work, important client/security tasks, and low-priority noise before composing the report.",
    ],
    "heldout_email13": [
        "Pay special attention to monitoring-style messages near the end of the inbox; a correlated alert still needs explicit coverage.",
        "Late-position alert emails may be operationally central. Include them and link them when they belong to the outage.",
    ],
    "heldout_actionability": [
        "The final report should let another engineer take over immediately: clear severity, owner-facing rationale, and next action.",
        "Write a concise but actionable handoff that preserves both the incident grouping and complete per-email coverage.",
    ],
}


TASK16_REWARD_RUBRIC = {
    "required_report_schema": "incident_groups_v1",
    "expected_incident_groups": [
        {
            "id": "production_database_incident",
            "emails": ["email_01", "email_13"],
            "priority": "p0",
        }
    ],
    "expected_priorities": {
        "email_01": "p0",
        "email_13": "p0",
        "email_05": "p1",
        "email_08": "p1",
        "email_10": "p2",
        "email_11": "p4",
    },
    "minimum_email_coverage": 10,
    "train_shaping_minimum_email_coverage": 10,
    "final_grader_minimum_email_coverage": 13,
}


READ_EXPLICIT_PREFIX = """The emails for this task are files in the workspace:
- inbox/email_01.txt through inbox/email_13.txt

You must use the read tool to inspect those inbox files. Do not search memory,
web, config, or unrelated sources for the emails. After reading the inbox files,
write the final triage artifact to triage_report.md.
"""


def _read_explicit_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if "inbox/email_01.txt" in prompt and "triage_report.md" in prompt:
        return prompt
    return f"{READ_EXPLICIT_PREFIX}\nAdditional triage instruction:\n{prompt}"


def _read_explicit_task(task: dict) -> dict:
    out = dict(task)
    out["prompt"] = _read_explicit_prompt(str(out["prompt"]))
    return out


def _row_with_group(
    task: dict,
    repeat_idx: int,
    group: str,
    reward_rubric: dict | None = None,
    workspace_files: list[dict] | None = None,
    instance_id: str | None = None,
) -> dict:
    row = build_verl_row(task, repeat_idx=repeat_idx)
    row["extra_info"]["prompt_group"] = group
    row["extra_info"]["reward_rubric"] = reward_rubric or TASK16_REWARD_RUBRIC
    if workspace_files:
        row["extra_info"]["workspace_files"] = workspace_files
    if instance_id:
        row["extra_info"]["instance_id"] = instance_id
        row["extra_info"]["synthetic_instance"] = True
    return row


SYNTHETIC_TASK16_PROMPTS = [
    "Triage this inbox and write `triage_report.md`. Preserve a per-email evidence trail: email id, priority, category, concrete clue, and recommended action.",
    "Process the inbox as an operator handoff. Link related outage and alert emails, cover every email id, and write the final result to `triage_report.md`.",
    "Build an evidence table from the inbox before summarizing. The final `triage_report.md` must make each priority traceable to real sender, subject, deadline, customer, or incident evidence.",
    "Do not rely on a memorized answer. Read the current inbox files, bind each email id to its actual content, and produce `triage_report.md`.",
]


SYNTHETIC_TRAIN_CASES = [
    ("AsterBank", "checkout", "postgres primary", "payment failures", "Sentinel", "p95 latency", "NimbusMart", "$1.8M renewal", "Okta token rotation", "May 3 17:00 UTC", "billing export freeze"),
    ("Northstar Health", "patient portal", "redis cluster", "login timeouts", "Watchtower", "5xx spike", "Cobalt Clinic", "$2.4M expansion", "VPN certificate rotation", "May 4 09:00 UTC", "audit log migration"),
    ("HelioPay", "card authorization", "risk service", "declined transactions", "PulseOps", "queue depth", "Vertex Retail", "$3.1M launch", "database password rotation", "May 5 12:00 UTC", "fraud review endpoint"),
    ("Atlas Freight", "shipment tracking", "search index", "blank tracking pages", "SignalDesk", "error budget burn", "Harbor Foods", "$950K contract", "SAML key rollover", "May 6 18:00 UTC", "carrier webhook patch"),
    ("BlueRiver", "mobile API", "gateway pods", "timeouts", "Redline", "API latency", "Summit Bank", "$5M migration", "production secret rotation", "May 7 10:00 UTC", "identity cache fix"),
    ("Mercury Travel", "booking flow", "inventory database", "failed reservations", "PagerLens", "db connection saturation", "Orchid Hotels", "$1.2M rollout", "MFA enforcement deadline", "May 8 16:00 UTC", "fare rules parser"),
    ("Acme Robotics", "device cloud", "mqtt broker", "offline devices", "Monarch", "broker reconnect storm", "IronWorks", "$2.7M fleet deal", "SSH key rotation", "May 9 08:00 UTC", "firmware rollout gate"),
    ("Pine Labs", "merchant dashboard", "analytics warehouse", "missing settlement data", "SentryWatch", "warehouse lag", "UrbanCart", "$1.5M renewal", "PCI evidence upload", "May 10 15:00 UTC", "settlement reconciliation"),
    ("Zenith Media", "ad delivery", "campaign cache", "overspend risk", "OpsBeacon", "cache miss storm", "BrightAds", "$800K quarter plan", "IAM access review", "May 11 11:00 UTC", "campaign budget lock"),
    ("Riverstone Energy", "meter ingestion", "kafka topic", "delayed readings", "MetricFox", "consumer lag", "GridCo", "$4.2M enterprise deal", "service account cleanup", "May 12 14:00 UTC", "meter replay job"),
    ("NovaLearn", "classroom video", "turn server", "session drops", "AlertNest", "packet loss", "EduPrime", "$1.1M school district", "FERPA access audit", "May 13 19:00 UTC", "video token refresh"),
    ("Canyon Insurance", "claims upload", "object storage", "upload failures", "Watchline", "storage 503s", "PolicyPro", "$2.0M renewal", "encryption key rotation", "May 14 07:00 UTC", "claims attachment validator"),
    ("OrbitTel", "subscriber portal", "auth database", "password reset failures", "PingScope", "auth 5xx", "MetroFiber", "$6M expansion", "SOC2 control evidence", "May 15 13:00 UTC", "reset-token review"),
    ("Lumen Games", "matchmaking", "session service", "players stuck in queue", "GameOps", "match latency", "ArcadePlus", "$900K launch", "admin role review", "May 16 21:00 UTC", "region failover config"),
    ("Evergreen Retail", "returns portal", "rules engine", "refund errors", "TraceKit", "exception burst", "MegaOutlet", "$1.6M renewal", "vendor API key rotation", "May 17 10:00 UTC", "refund policy deploy"),
    ("Silverline SaaS", "admin console", "permissions service", "role update failures", "NodeWatch", "permission write errors", "DataForge", "$2.9M contract", "SCIM token rotation", "May 18 18:00 UTC", "permissions review"),
    ("Coral Foods", "delivery dispatch", "routing database", "late driver assignments", "OpsRadar", "routing latency", "FreshBox", "$1.3M regional rollout", "payment key rotation", "May 19 09:30 UTC", "driver assignment patch"),
    ("Peak Finance", "loan decisioning", "feature store", "stale risk scores", "MetricGuard", "feature freshness", "CreditUnion One", "$3.5M pilot", "privileged access review", "May 20 17:30 UTC", "risk model config"),
    ("Aurora Logistics", "warehouse scanner", "sync API", "inventory mismatch", "AlertGrid", "sync error rate", "NorthDepot", "$1.0M expansion", "scanner cert rotation", "May 21 06:00 UTC", "inventory conflict resolver"),
    ("VegaCloud", "tenant provisioning", "control plane", "provisioning stuck", "PulseMeter", "control plane saturation", "AlphaApps", "$4.8M migration", "breakglass account review", "May 22 20:00 UTC", "tenant quota fix"),
]

SYNTHETIC_VAL_CASES = [
    ("MapleBank", "wire transfer", "ledger database", "transfer failures", "NightWatch", "ledger write latency", "Prime Treasury", "$2.6M treasury rollout", "HSM key rotation", "May 23 11:00 UTC", "wire approval patch"),
    ("Skyline Retail", "coupon checkout", "promotion service", "discount errors", "SignalFox", "promotion 5xx", "ShopHub", "$1.4M campaign launch", "SSO cert rollover", "May 24 15:00 UTC", "coupon rules review"),
    ("MedAtlas", "lab results portal", "document store", "missing lab PDFs", "CarePulse", "document fetch failures", "Regional Clinic", "$3.2M deployment", "PHI access audit", "May 25 08:00 UTC", "pdf retrieval fix"),
    ("ForgeCloud", "build runners", "scheduler service", "stalled CI jobs", "BuildWatch", "runner queue saturation", "EnterpriseTools", "$1.7M migration", "runner token rotation", "May 26 22:00 UTC", "scheduler fairness change"),
    ("HarborPay", "invoice sync", "webhook processor", "delayed invoices", "HookMeter", "webhook retry storm", "OceanFoods", "$900K renewal", "bank API credential rotation", "May 27 13:30 UTC", "invoice retry limiter"),
]


def _email(path_id: int, sender: str, subject: str, body: str) -> dict:
    return {
        "path": f"inbox/email_{path_id:02d}.txt",
        "content": f"From: {sender}\nSubject: {subject}\n\n{body}\n",
    }


def _synthetic_instance(case_idx: int, case: tuple[str, ...]) -> tuple[list[dict], dict]:
    (
        company,
        surface,
        component,
        impact,
        monitor,
        alert_signal,
        client,
        contract,
        security_item,
        deadline,
        release_blocker,
    ) = case
    workspace_files = [
        _email(1, "incident-manager@example.com", f"P0: {company} {surface} outage", f"{company}'s {surface} is failing because the {component} is unhealthy. Customer-facing users report {impact}. War room is active and engineering needs immediate incident response."),
        _email(2, "marketing@example.com", "Draft webinar blurb for next week", "Please review the draft webinar announcement by Wednesday. This is useful but not urgent."),
        _email(3, "dependabot@example.com", "Dependency update passed CI", "Automated dependency update for a minor library version. CI is green and this can be reviewed later."),
        _email(4, "people-ops@example.com", "Benefits enrollment reminder", "Reminder to finish benefits enrollment before the end of the month."),
        _email(5, f"account-owner-{client.lower().replace(' ', '-')}@example.com", f"{client} escalation blocks {contract}", f"{client} cannot proceed with the planned rollout until the team answers technical questions today. The thread explicitly references {contract} and customer executive visibility."),
        _email(6, "calendar@example.com", "Weekly team sync moved", "The weekly internal team sync moved by thirty minutes. No customer or production impact."),
        _email(7, "manager@example.com", "Quarterly self assessment due", "Please submit your self assessment this week. Important internally, but it does not block production or customers today."),
        _email(8, "security@example.com", f"Security deadline: {security_item}", f"The {security_item} must be completed by {deadline}. Missing the deadline creates compliance and production access risk."),
        _email(9, "newsletter@example.com", "Industry digest", "A product newsletter with articles and conference links. No operational action required."),
        _email(10, "review-bot@example.com", f"Review needed: {release_blocker}", f"The {release_blocker} pull request is blocking the release train. Engineering review is needed before deployment can continue."),
        _email(11, "promo@example.net", "Limited time vendor discount", "Promotional vendor email. It does not relate to customers, security, release, or production."),
        _email(12, "finance@example.com", "Budget notes for planning", "Finance shared budget notes for next planning cycle. Review when convenient."),
        _email(13, f"{monitor.lower()}@monitoring.example.com", f"Alert: {surface} {alert_signal}", f"{monitor} reports {alert_signal} on the same {surface} path while the {component} incident is active. This alert corroborates the outage and should be linked to email_01."),
    ]
    bindings = {
        "email_01": [company.lower(), surface.lower(), component.lower(), "outage"],
        "email_05": [client.lower(), contract.lower().replace("$", ""), "customer"],
        "email_08": [security_item.lower(), deadline.lower(), "security"],
        "email_10": [release_blocker.lower(), "release", "review"],
        "email_11": ["promotional", "vendor"],
        "email_13": [monitor.lower(), alert_signal.lower(), surface.lower(), "alert"],
    }
    rubric = dict(TASK16_REWARD_RUBRIC)
    rubric["expected_incident_groups"] = [
        {
            "id": f"{company.lower().replace(' ', '_')}_{surface.lower().replace(' ', '_')}_incident",
            "emails": ["email_01", "email_13"],
            "priority": "p0",
            "required_clues": [company.lower(), surface.lower(), component.lower(), alert_signal.lower()],
        }
    ]
    rubric["expected_bindings"] = {
        email_id: {"required_any": clues, "min_matches": 2}
        for email_id, clues in bindings.items()
    }
    rubric["expected_priorities"] = {
        "email_01": "p0",
        "email_13": "p0",
        "email_05": "p1",
        "email_08": "p1",
        "email_10": "p2",
        "email_11": "p4",
    }
    rubric["synthetic_case"] = {
        "company": company,
        "surface": surface,
        "client": client,
        "monitor": monitor,
    }
    return workspace_files, rubric


def _synthetic_rows(
    canonical: dict,
    cases: list[tuple[str, ...]],
    repeat_offset: int = 4000,
    group: str = "synthetic_instance",
    instance_prefix: str = "task16_synth",
) -> list[dict]:
    rows: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        workspace_files, rubric = _synthetic_instance(idx, case)
        task = dict(canonical)
        task["prompt"] = SYNTHETIC_TASK16_PROMPTS[(idx - 1) % len(SYNTHETIC_TASK16_PROMPTS)]
        rows.append(
            _row_with_group(
                task,
                repeat_idx=repeat_offset + idx,
                group=group,
                reward_rubric=rubric,
                workspace_files=workspace_files,
                instance_id=f"{instance_prefix}_{idx:02d}",
            )
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet_if_available(path: Path, rows: list[dict]) -> bool:
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except ImportError as exc:
        print(f"skip_parquet={path} reason={exc}")
        return False


def _heldout_val_entries(limit: int) -> list[tuple[str, str]]:
    entries = [
        (group, prompt)
        for group, prompts in HELDOUT_VAL_PROMPT_GROUPS.items()
        for prompt in prompts
    ]
    if limit <= 0:
        return entries
    return entries[:limit]


def _limited_train_entries(
    prompt_entries: list[tuple[str, str]],
    max_train_rows: int,
) -> list[tuple[str, str]]:
    if max_train_rows <= 0:
        return prompt_entries
    variant_limit = max(0, max_train_rows - 1)
    if variant_limit >= len(prompt_entries):
        return prompt_entries

    group_order = ["base", *TARGETED_PROMPT_GROUPS.keys()]
    buckets = {
        group: [prompt for entry_group, prompt in prompt_entries if entry_group == group]
        for group in group_order
    }
    selected: list[tuple[str, str]] = []
    cursor = 0
    while len(selected) < variant_limit:
        made_progress = False
        for group in group_order:
            prompts = buckets[group]
            if cursor < len(prompts):
                selected.append((group, prompts[cursor]))
                made_progress = True
                if len(selected) >= variant_limit:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


SMALL_TRAIN_GROUP_ORDER = [
    "base",
    "evidence_preservation",
    "coverage_before_summary",
    "anti_template",
    "priority_audit",
    "coverage_audit",
    "incident_linkage",
    "email13_priority",
    "priority_propagation",
    "bigclient_weighting",
    "security_weighting",
    "report_schema_incident_groups",
    "report_schema_priority_fields",
]

TINY_TRAIN_GROUP_ORDER = [
    "evidence_preservation",
    "coverage_before_summary",
    "anti_template",
    "priority_audit",
    "coverage_audit",
    "incident_linkage",
    "email13_priority",
    "report_schema_priority_fields",
]


def _focused_rows(
    canonical: dict,
    prompt_entries: list[tuple[str, str]],
    group_order: list[str],
    target_count: int,
    repeat_offset: int,
) -> list[dict]:
    rows = [_row_with_group(canonical, repeat_idx=repeat_offset, group="canonical")]
    by_group: dict[str, list[str]] = {
        group: [prompt for entry_group, prompt in prompt_entries if entry_group == group]
        for group in group_order
    }

    cursor = 0
    while len(rows) < target_count:
        made_progress = False
        for group in group_order:
            prompts = by_group.get(group, [])
            if cursor >= len(prompts):
                continue
            task = dict(canonical)
            task["prompt"] = prompts[cursor]
            rows.append(_row_with_group(task, repeat_idx=repeat_offset + len(rows), group=group))
            made_progress = True
            if len(rows) >= target_count:
                break
        if not made_progress:
            break
        cursor += 1

    if len(rows) != target_count:
        raise SystemExit(f"Could only build {len(rows)} focused rows, wanted {target_count}")
    return rows


def _read_explicit_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        new_row = dict(row)
        prompt = new_row.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            new_prompt = [dict(item) for item in prompt]
            new_prompt[0]["content"] = _read_explicit_prompt(str(new_prompt[0].get("content", "")))
            new_row["prompt"] = new_prompt
        elif hasattr(prompt, "tolist"):
            prompt_list = prompt.tolist()
            if prompt_list and isinstance(prompt_list[0], dict):
                new_prompt = [dict(item) for item in prompt_list]
                new_prompt[0]["content"] = _read_explicit_prompt(str(new_prompt[0].get("content", "")))
                new_row["prompt"] = new_prompt
        extra_info = dict(new_row.get("extra_info") or {})
        group = str(extra_info.get("prompt_group", ""))
        if not group.endswith("_readexplicit"):
            extra_info["prompt_group"] = f"{group}_readexplicit" if group else "readexplicit"
        new_row["extra_info"] = extra_info
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task_16 RL prompt variants")
    parser.add_argument("--tasks-dir", type=Path, default=Path("pinchbench_tasks"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rl/data/prompts_task16_variants"),
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=11,
        help="Number of held-out validation wordings to emit. Use 20 for the full held-out set.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap train rows including canonical. Use 0 to emit the full prompt pool.",
    )
    args = parser.parse_args()

    task_path = resolve_task_markdown_path(args.tasks_dir, "task_16_email_triage")
    if not task_path.exists():
        raise SystemExit(f"Task file not found: {task_path}")

    canonical = parse_task_file(task_path)
    rows = [_row_with_group(canonical, repeat_idx=0, group="canonical")]

    variant_tasks: list[dict] = []
    prompt_entries: list[tuple[str, str]] = []
    prompt_entries.extend(("base", p) for p in BASE_PROMPTS)
    for group, prompts in TARGETED_PROMPT_GROUPS.items():
        prompt_entries.extend((group, p) for p in prompts)

    train_entries = _limited_train_entries(prompt_entries, args.max_train_rows)

    for i, (group, prompt) in enumerate(train_entries, start=1):
        task = dict(canonical)
        task["prompt"] = prompt
        variant_tasks.append(task)
        rows.append(_row_with_group(task, repeat_idx=i, group=group))
    synthetic_rows = _synthetic_rows(
        canonical,
        SYNTHETIC_TRAIN_CASES,
        repeat_offset=4000,
        group="synthetic_instance",
        instance_prefix="task16_synth",
    )
    synthetic_val_rows = _synthetic_rows(
        canonical,
        SYNTHETIC_VAL_CASES,
        repeat_offset=5000,
        group="synthetic_val_instance",
        instance_prefix="task16_synth_val",
    )
    rows.extend(synthetic_rows)

    val_entries = _heldout_val_entries(args.val_count)
    train_prompt_set = {canonical["prompt"], *(prompt for _, prompt in train_entries)}
    val_prompt_set = {prompt for _, prompt in val_entries}
    overlap = train_prompt_set & val_prompt_set
    if overlap:
        sample = " ".join(next(iter(overlap)).split())[:160]
        raise SystemExit(f"held-out val prompt overlaps train prompt: {sample}")

    val_rows = []
    for i, (group, prompt) in enumerate(val_entries):
        task = dict(canonical)
        task["prompt"] = prompt
        val_rows.append(_row_with_group(task, repeat_idx=999 + i, group=group))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl_path = args.output_dir / "train.jsonl"
    train_small_jsonl_path = args.output_dir / "train_small.jsonl"
    train_tiny_jsonl_path = args.output_dir / "train_tiny.jsonl"
    train_synth_jsonl_path = args.output_dir / "train_synth20.jsonl"
    train_stage2_jsonl_path = args.output_dir / "train_stage2_balanced.jsonl"
    train_readexplicit_jsonl_path = args.output_dir / "train_canonical32_readexplicit.jsonl"
    val_synth_jsonl_path = args.output_dir / "val_synth5.jsonl"
    val_readexplicit_jsonl_path = args.output_dir / "val_canonical5_readexplicit.jsonl"
    val_jsonl_path = args.output_dir / "val.jsonl"
    train_path = args.output_dir / "train.parquet"
    train_small_path = args.output_dir / "train_small.parquet"
    train_tiny_path = args.output_dir / "train_tiny.parquet"
    train_synth_path = args.output_dir / "train_synth20.parquet"
    train_stage2_path = args.output_dir / "train_stage2_balanced.parquet"
    train_readexplicit_path = args.output_dir / "train_canonical32_readexplicit.parquet"
    val_synth_path = args.output_dir / "val_synth5.parquet"
    val_readexplicit_path = args.output_dir / "val_canonical5_readexplicit.parquet"
    val_path = args.output_dir / "val.parquet"
    train_small_rows = _focused_rows(canonical, prompt_entries, SMALL_TRAIN_GROUP_ORDER, 32, 2000)
    train_tiny_rows = _focused_rows(canonical, prompt_entries, TINY_TRAIN_GROUP_ORDER, 16, 3000)
    train_stage2_rows = train_small_rows[:12] + synthetic_rows
    train_readexplicit_rows = _read_explicit_rows(train_small_rows)
    val_readexplicit_rows = _read_explicit_rows(val_rows[:5])
    _write_jsonl(train_jsonl_path, rows)
    _write_jsonl(train_small_jsonl_path, train_small_rows)
    _write_jsonl(train_tiny_jsonl_path, train_tiny_rows)
    _write_jsonl(train_synth_jsonl_path, synthetic_rows)
    _write_jsonl(train_stage2_jsonl_path, train_stage2_rows)
    _write_jsonl(train_readexplicit_jsonl_path, train_readexplicit_rows)
    _write_jsonl(val_synth_jsonl_path, synthetic_val_rows)
    _write_jsonl(val_readexplicit_jsonl_path, val_readexplicit_rows)
    _write_jsonl(val_jsonl_path, val_rows)
    train_parquet = _write_parquet_if_available(train_path, rows)
    train_small_parquet = _write_parquet_if_available(train_small_path, train_small_rows)
    train_tiny_parquet = _write_parquet_if_available(train_tiny_path, train_tiny_rows)
    train_synth_parquet = _write_parquet_if_available(train_synth_path, synthetic_rows)
    train_stage2_parquet = _write_parquet_if_available(train_stage2_path, train_stage2_rows)
    train_readexplicit_parquet = _write_parquet_if_available(train_readexplicit_path, train_readexplicit_rows)
    val_synth_parquet = _write_parquet_if_available(val_synth_path, synthetic_val_rows)
    val_readexplicit_parquet = _write_parquet_if_available(val_readexplicit_path, val_readexplicit_rows)
    val_parquet = _write_parquet_if_available(val_path, val_rows)

    print(f"Wrote {len(rows)} train prompts to {train_jsonl_path}")
    print(f"Wrote {len(train_small_rows)} focused train prompts to {train_small_jsonl_path}")
    print(f"Wrote {len(train_tiny_rows)} focused smoke prompts to {train_tiny_jsonl_path}")
    print(f"Wrote {len(synthetic_rows)} synthetic train instances to {train_synth_jsonl_path}")
    print(f"Wrote {len(train_stage2_rows)} stage2 balanced train rows to {train_stage2_jsonl_path}")
    print(f"Wrote {len(train_readexplicit_rows)} read-explicit canonical train rows to {train_readexplicit_jsonl_path}")
    print(f"Wrote {len(synthetic_val_rows)} synthetic val instances to {val_synth_jsonl_path}")
    print(f"Wrote {len(val_readexplicit_rows)} read-explicit canonical val rows to {val_readexplicit_jsonl_path}")
    print(f"Wrote {len(val_rows)} val prompts to {val_jsonl_path}")
    print(
        "parquet_written "
        f"train={train_parquet} "
        f"train_small={train_small_parquet} "
        f"train_tiny={train_tiny_parquet} "
        f"train_synth20={train_synth_parquet} "
        f"train_stage2_balanced={train_stage2_parquet} "
        f"train_canonical32_readexplicit={train_readexplicit_parquet} "
        f"val_synth5={val_synth_parquet} "
        f"val_canonical5_readexplicit={val_readexplicit_parquet} "
        f"val={val_parquet}"
    )
    print(f"Canonical task_id: {canonical['task_id']}")
    print(f"Variant prompts: {len(variant_tasks)}")
    print(f"Synthetic task16-style instances: {len(synthetic_rows)}")
    print(f"Synthetic held-out val instances: {len(synthetic_val_rows)}")
    print(f"Max train rows: {args.max_train_rows or 'all'}")
    print(f"Base prompts: {len(BASE_PROMPTS)}")
    print(f"Targeted prompts: {sum(len(v) for v in TARGETED_PROMPT_GROUPS.values())}")
    print(f"Held-out val prompt overlap with train: {len(overlap)}")
    print(f"Reward minimum_email_coverage: {TASK16_REWARD_RUBRIC['minimum_email_coverage']}")
    for group, prompts in TARGETED_PROMPT_GROUPS.items():
        print(f"  group[{group}] = {len(prompts)}")
    preview_prompts = [canonical["prompt"]] + [p for _, p in prompt_entries[:4]]
    for idx, prompt in enumerate(preview_prompts):
        preview = " ".join(prompt.split())[:120]
        print(f"  [{idx}] {preview}...")


if __name__ == "__main__":
    main()
