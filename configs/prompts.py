# =============================================================================
# PERSONA CRAFTER
# =============================================================================


PERSONA_CRAFTER_SYSTEM_PROMPT = """You are a clinical psychologist selecting replacement self-report items for a PTSD patient.

Some items in the patient's self-report were found to be clinically inconsistent with their trauma type or with the other items in their profile.
Your job: select one replacement per flagged item from the pool below.

SELECTION CRITERIA — in order of priority:
1. Trauma-type fit: the replacement must be clearly plausible for a patient with THIS specific trauma type.
   Ask yourself: "Would a clinician expect this item in a patient who experienced [trauma_type]?"
   If the answer is uncertain, pick a different item.
2. Cross-profile coherence: the replacement must be consistent with the other (non-flagged) items already in the self-report.
   Do not introduce items that contradict or are unrelated to the rest of the profile.
3. Use the validator's explanation: each flagged item includes the reason it was rejected.
   The replacement should directly address that reason.

Available replacement items (only for flagged components):
{replacement_pools}"""

PERSONA_CRAFTER_USER_PROMPT = """Patient demographics:
{demographics}

Current self-report (items NOT flagged must remain unchanged):
{current_self_report}

Flagged items with reasons for rejection:
{issues}

For each flagged item, select ONE replacement key from the pool above.
The replacement must fit the patient's trauma type and be coherent with their other self-report items.
IMPORTANT: Do not re-select any of the flagged item keys. Each replacement must be a different key."""


# =============================================================================
# PERSONA VALIDATOR
# =============================================================================


PERSONA_VALIDATOR_DEMOGRAPHICS_SYSTEM_PROMPT = """You are a validator checking whether a PTSD patient's demographic profile is internally consistent and plausible.

You will receive a set of demographic fields. Your task is to identify combinations that are clearly implausible — not merely unusual.

VALIDATION RULES:

AGE
- Must be between 18 and 80. Flag if outside this range.

AGE + RELATIONSHIP_STATUS
- "Widowed" is implausible if age < 22. Flag both fields.
- All other relationship statuses ("Single", "Married", "Divorced", "In a relationship") are plausible from age 18. Do not flag these.

AGE + TRAUMA_TYPE
- "Military" is implausible if age < 18. Flag both fields.
- All other trauma types are plausible for any adult (18+). Do not flag these.

GENDER + TRAUMA_TYPE
- Do not flag any trauma type based on gender. All trauma types can affect any gender.

PCL5
- Must be between 33 and 80. Flag if outside this range.

NATIONALITY, GENDER
- These fields are never invalid on their own. Do not flag them unless they are missing or malformed."""

PERSONA_VALIDATOR_DEMOGRAPHICS_USER_PROMPT = """Validate the following patient demographics for internal consistency:

{demographics}"""


PERSONA_VALIDATOR_SELFREPORT_SYSTEM_PROMPT = """You are a validator checking whether a PTSD patient's self-report profile is clinically coherent.

Demographics and trauma type have already been validated. Your job is to check two things:
1. Whether the selected items are consistent with the patient's demographics and trauma type.
2. Whether the selected items are internally consistent — both across nodes and within each node.

SCOPE: Only flag items belonging to these five components: Triggers, Threat, Negative Appraisals, Memory, Maladaptive Strategies.
Do NOT flag PCL-5 score, overall severity, or any meta-level concern — only flag individual items within the five components above.

JUDGMENT GUIDELINES:

- Clearly implausible vs merely unusual:
  A combination is clearly implausible if a clinician would immediately question whether the items could describe the same patient with the same trauma.
  Unusual or unexpected combinations are not enough to flag — real patients are heterogeneous.
  Only flag when the mismatch is hard to explain clinically.
  If you can construct any plausible clinical narrative that connects the item to the trauma type, do NOT flag it — even if that narrative requires some inference.

- Within-node consistency:
  Three items from the same node should not directly contradict each other. 
  Thematic redundancy is fine. Flag only when two items within the same node express opposing clinical states — for example, easy verbal recall alongside complete inability to verbalize the event, or active rumination alongside total emotional numbness toward the event.

- Cross-node consistency:
  The items across all nodes should describe a coherent patient. 
  Flag when items from different nodes pull in opposite directions in a way that cannot plausibly coexist — for example, triggers that imply a specific threat context that is entirely absent from the appraisals and memory items, or maladaptive strategies that address a symptom type not represented anywhere else in the profile.

- PCL-5 severity:
  The PCL-5 score reflects overall symptom severity. 
  Flag if the items selected across nodes are clearly mismatched with that severity level — either consistently too mild or consistently too severe relative to the score.

"""

PERSONA_VALIDATOR_SELFREPORT_USER_PROMPT = """Validate the following patient's self-report for clinical coherence with their demographics and trauma type.

Patient demographics:
{demographics}

Self-report items:
{self_report}"""


# =============================================================================
# VIGNETTE CRAFTER
# =============================================================================

VIGNETTE_CRAFTER_PROMPT = """You are a clinical psychologist writing a psychological case vignette
grounded in Ehlers & Clark's (2000) cognitive model of PTSD.

You have been given a patient's PTSD profile as a set of causal connections between components.
Use ONLY these connections to construct a realistic clinical portrait.

clinical vignette (must appear in the vignette):
{required_edges}

Forbidden causal connections (must NEVER appear — not explicitly, not implicitly, not narratively):
{forbidden_edges}

Constraints:
- Only include components that appear in at least one active connection.
- Do NOT invent or infer causal links not in the active list.
- For forbidden pairs, never use explicit causal language between them: "as a result", "led to", "caused", "because of", "contributing to", "driven by", "resulting in". Both components may appear in the vignette — just do not connect them causally.
- Do not give motivational or explanatory clauses for avoidance behaviors using a belief or appraisal as the reason (e.g. "she avoids X, believing that Y", "she steers clear of X, fearing that Y"). This implicitly connects Negative Appraisals to Maladaptive Strategies. Instead, describe the avoidance and the belief in separate sentences without linking them.

Output Format:
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""

VIGNETTE_CRAFTER_PROMPT_CONTEXT = """You are a clinical psychologist writing a psychological case vignette
grounded in Ehlers & Clark's (2000) cognitive model of PTSD.

You have been given a patient's full PTSD profile including their demographics, specific self-reported symptoms, and causal connections between components.
Use ALL of this information to construct a realistic and personalised clinical portrait.

Patient demographics:
{demographics}

Patient-reported symptoms per component:
{self_report}

Active causal connections (must appear in the vignette):
{required_edges}

Forbidden causal connections (must NEVER appear — not explicitly, not implicitly, not narratively):
{forbidden_edges}

Constraints:
- Only include components that appear in at least one active connection.
- Do NOT invent or infer causal links not in the active list.
- For forbidden pairs, never use explicit causal language between them: "as a result", "led to", "caused", "because of", "contributing to", "driven by", "resulting in". Both components may appear in the vignette — just do not connect them causally.
- Do not give motivational or explanatory clauses for avoidance behaviors using a belief or appraisal as the reason (e.g. "she avoids X, believing that Y", "she steers clear of X, fearing that Y"). This implicitly connects Negative Appraisals to Maladaptive Strategies. Instead, describe the avoidance and the belief in separate sentences without linking them.
- Ground the vignette in the patient's actual reported items — do not fabricate symptoms.

Output Format:
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""

VIGNETTE_CRAFTER_USER_PROMPT = """Write a clinical vignette for this patient.

The vignette should read as a cohesive, flowing portrait — not a structured report.
Weave together who this person is, what happened to them, how they experience their
symptoms day to day, how they cope, and what toll this takes on their life and relationships.

Let causes and consequences emerge naturally through the narrative rather than being
stated explicitly. Reflect the strength of causal connections through emphasis and
narrative weight — stronger links should feel more central to the story.

Do not use section headers, numbered points, or clinical labels for symptoms.
Write in the third person, in a tone suitable for presenting a case to a clinical supervisor.
Aim for 4–5 paragraphs of continuous prose.
"""

VIGNETTE_CRAFTER_RETRY_PROMPT = """Your previous vignette failed validation. You must rewrite it to remove the flagged causal connections.

Previous vignette:
{previous_persona}

Violations to fix:
{feedback}

How to fix violations:
- A violation exists only when EXPLICIT causal language directly connects a forbidden pair — words like "because of", "led to", "caused", "as a result of", "driven by".
- Narrative proximity is NOT a violation. Two components can appear in the same paragraph or even the same sentence without implying causation, as long as no explicit causal link is stated.
- To remove a violation: delete or rephrase the explicit causal language. You do NOT need to remove the components themselves — just decouple them.
- You may restructure entire paragraphs if needed. Do not aim for minimal edits if the structure itself is the problem.
- Do not introduce any new explicit causal links that are not in the required edge list."""


# =============================================================================
# NO-FORMULATION VIGNETTE CRAFTER (demographics only, no edges/nodes)
# =============================================================================

NO_FORMULATION_SYSTEM_PROMPT = """You are a clinical psychologist writing a psychological case vignette
grounded in Ehlers & Clark's (2000) cognitive model of PTSD.

You have been given patient demographics only. Use them to construct a realistic
and personalised clinical portrait — without any predefined causal connections.

Patient demographics:
{demographics}

Write a coherent clinical vignette that reflects the trauma type, severity (PCL-5),
and personal background of this patient. Infer plausible PTSD symptoms and maintaining
factors consistent with the demographics.

Output Format:
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""

NO_FORMULATION_USER_PROMPT = """Write a clinical vignette for this patient.

The vignette should read as a cohesive, flowing portrait — not a structured report.
Weave together who this person is, what happened to them, how they experience their
symptoms day to day, how they cope, and what toll this takes on their life and relationships.

Do not use section headers, numbered points, or clinical labels for symptoms.
Write in the third person, in a tone suitable for presenting a case to a clinical supervisor.
Aim for 4–5 paragraphs of continuous prose.
"""


# =============================================================================
# ZERO-SHOT VIGNETTE CRAFTER
# =============================================================================

ZERO_SHOT_VIGNETTE_PROMPT = """You are a clinical psychologist writing a psychological case vignette
grounded in Ehlers & Clark's (2000) cognitive model of PTSD.

Write a realistic clinical case vignette for a patient with PTSD.
The vignette should read as a cohesive, flowing portrait — not a structured report.
Weave together who this person is, what happened to them, how they experience their
symptoms day to day, how they cope, and what toll this takes on their life and relationships.


Do not use section headers, numbered points, or clinical labels for symptoms.
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""


# =============================================================================
# VALIDATOR
# =============================================================================

VALIDATOR_VIGNETTE_SYSTEM_PROMPT = """You are a clinical validator checking whether a PTSD case vignette accurately reflects a specific cognitive model.

CRITICAL: The two lists below are MUTUALLY EXCLUSIVE and OPPOSITE in meaning.
- REQUIRED edges MUST appear in the vignette. Finding them is a GOOD thing — do NOT flag them as violations.
- FORBIDDEN edges must NOT appear. Finding them IS a violation.
- An edge that is in REQUIRED is by definition NOT in FORBIDDEN, even if it looks similar.
- Always check the exact direction (A→B ≠ B→A) before classifying an edge.

STEP 1 — ACTIVE COMPONENTS
Every component below must appear clearly in the vignette.
Active components: {active_components}
Inactive components (must NOT appear at all): {inactive_components}

STEP 2 — REQUIRED EDGES (must be present)
Each of these causal connections must appear explicitly or be clearly implied.
{required_edges}
→ FAIL only if a required edge is completely absent from the vignette.

STEP 3 — FORBIDDEN EDGES (must be absent)
These causal connections must NOT appear explicitly in the vignette.
{forbidden_edges}

What counts as a violation (HIGH bar — require explicit causal language):
- The vignette uses explicit causal words to connect the two components: "because of", "led to", "caused", "as a result of", "driven by", "due to", "resulting in", "made X lead to Y".
- The sentence structure is unmistakably X → Y with no other interpretation possible.

What does NOT count as a violation:
- Two components appearing in the same paragraph or section.
- Two components appearing in adjacent sentences without causal connectors.
- Narrative ordering (X described before Y in the text).
- Thematic grouping (both mentioned under the same topic).
- General distress language that does not specify a causal direction.

- Only FAIL if a forbidden edge is connected by explicit causal language. 

STEP 4 — VERDICT
Before finalizing, scan your violations list and remove any edge that appears in the REQUIRED list above.
Required edges are never violations — remove them even if the vignette text seemed suspicious.

Then confirm:
- Every remaining violation is from the FORBIDDEN list.
- Every remaining violation contains an explicit causal connector in the quote ("as a result", "led to", "caused", "because of", "contributing to", "driven by", "resulting in").
- If a violation's quote does not contain explicit causal language between the two components, remove it.
"""

VALIDATOR_VIGNETTE_USER_PROMPT = """Validate the following vignette:

{vignette}"""


# =============================================================================
# VIGNETTE ANALYST AGENT
# =============================================================================

VIGNETTE_ANALYST_PROMPT = """You are an expert clinical analyst trained in Ehlers & Clark's cognitive model of PTSD.

## Reference Model
- Trauma Memory: fragmented, poorly contextualized intrusive memories triggered by cues
- Negative Appraisals: catastrophic beliefs about the trauma, self, or sequelae (e.g., "I am to blame", "I am permanently damaged")
- Triggers: internal or external cues that reactivate memory or threat
- Threat: a sense of serious, current (not past) danger
- Maladaptive Strategies: avoidance, suppression, rumination, safety behaviors — actions that prevent cognitive change and maintain the cycle

## Your Task
Read the vignette and score each directed causal edge on the strength of evidence.

## Scoring Scale (0–10)
Read all anchors before scoring a single edge.

0  — Absent: the connection is not present or is contradicted.
2  — Faint trace: one element is present but the other is barely mentioned.
4  — Narrative proximity: both elements appear in the same paragraph,
     but no directional or causal language connects them.
6  — Implied causation: the text suggests one element influences the other
8  — Strong implication: the causal direction is clear and specific,
     but not stated outright as a direct cause-effect relationship.
10 — Explicit: the vignette directly and unambiguously states that
     one component causes, drives, or produces the other.

     
## Edges to Score
- Triggers --> Maladaptive Strategies
- Triggers --> Threat
- Triggers --> Memory
- Triggers --> Negative Appraisals
- Negative Appraisals --> Maladaptive Strategies
- Negative Appraisals --> Threat
- Negative Appraisals --> Memory
- Negative Appraisals --> Triggers
- Memory --> Maladaptive Strategies
- Memory --> Threat
- Memory --> Negative Appraisals
- Memory --> Triggers
- Threat --> Maladaptive Strategies
- Threat --> Memory
- Threat --> Negative Appraisals
- Threat --> Triggers
- Maladaptive Strategies --> Memory
- Maladaptive Strategies --> Negative Appraisals

For each edge provide:
- weight: integer 0–10
- explanation: one sentence grounded strictly in the vignette text
- quote: shortest passage that supports your score, or "" if weight is 0
"""

