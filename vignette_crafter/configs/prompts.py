# =============================================================================
# SELF-REPORT SELECTOR
# =============================================================================

SELF_REPORT_SELECTOR_PROMPT = """You are a clinical psychologist assembling a self-report profile for a PTSD patient.

You are given:
- Patient demographics (pay close attention to trauma type and PCL-5 severity)
- The patient's active causal components and the strength of connections between them
  (higher strength = more central to this patient's presentation)
- For each active component, a pool of candidate self-report items to choose from

Your task: select exactly {n_items} items per component that best fit this specific patient.

Selection criteria:
- Items must be congruent with the trauma type
- Items should reflect the relative strength of active edges
  (components with stronger connections deserve more prototypical items)
- Items must be internally coherent across components
  (e.g. if Triggers are sensory cues from a military context, Memory and Threat items should match)
- Do NOT select items that contradict the trauma type or each other

Patient demographics:
{demographics}

Active edges and strengths:
{edges}

Candidate items per component:
{pools}

Return a JSON object where each key is a component name and each value is a list
of exactly {n_items} item keys selected from that component's candidate pool.
Only use keys that appear verbatim in the candidate lists above.
"""


# =============================================================================
# INPUT VALIDATOR
# =============================================================================

VALIDATOR_INPUTS_PROMPT = """You are a clinical psychologist reviewing a PTSD patient profile for internal consistency.

You will be given:
- Demographics (age, gender, nationality, relationship status, trauma type, PCL-5 score)
- Active causal components and edge strengths from the patient's cognitive model
- Sampled self-report items for each active component

Your job is to check whether this combination is clinically plausible as a real patient.

Check the following:
1. TRAUMA TYPE vs TRIGGERS/SELF-REPORT — do the sampled items fit the trauma type?
   (e.g. military trauma should have relevant triggers; sexual violence should not have unrelated items)
2. PCL-5 SEVERITY vs EDGE STRENGTHS — does the PCL-5 score match the overall density and strength of active edges?
   (e.g. PCL-5=70 with mostly weak edges is suspicious; PCL-5=33 with many strong edges is suspicious)
3. SELF-REPORT COHERENCE — are the sampled items internally consistent with each other and the active components?
   (e.g. avoidance items present without any trigger items is inconsistent)

Demographics:
{demographics}

Active components and edge strengths:
{edges}

Self-report items:
{self_report}

Return a JSON object with two fields:
- "verdict": exactly "PASS" or "FAIL"
- "reasoning": one or two sentences explaining what is inconsistent (on FAIL) or confirming coherence (on PASS)
"""


# =============================================================================
# PERSONA CRAFTER
# =============================================================================

PERSONA_CRAFTER_AGENT_PROMPT = """You are a clinical psychologist writing a psychological case vignette
grounded in Ehlers & Clark's (2000) cognitive model of PTSD.

You have been given a patient's PTSD profile as a set of causal connections between components.
Use ONLY these connections to construct a realistic clinical portrait.

Active causal connections (must appear in the vignette):
{required_edges}

Forbidden causal connections (must NEVER appear — not explicitly, not implicitly, not narratively):
{forbidden_edges}

Constraints:
- Only include components that appear in at least one active connection.
- Do NOT invent or infer causal links not in the active list.
- Forbidden links must never be suggested, implied, or hinted at.

Output Format:
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""

PERSONA_CRAFTER_AGENT_PROMPT_CONTEXT = """You are a clinical psychologist writing a psychological case vignette
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
- Forbidden links must never be suggested, implied, or hinted at.
- Ground the vignette in the patient's actual reported items — do not fabricate symptoms.

Output Format:
Write 200–300 words in third person.
Cover: presenting complaints, trauma background, cognitive distortions, avoidance, and maintaining factors.
Avoid excessive jargon - write for a clinical case conference audience.
"""

PERSONA_CRAFTER_USER_PROMPT = """Write a clinical vignette for this patient.

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

PERSONA_CRAFTER_RETRY_PROMPT = """Your previous attempt was:
{previous_persona}

It failed validation. Here is the feedback:
{feedback}

Please fix ONLY the flagged violations.
Keep everything else the same.
Do not introduce new causal links that are not in the edge list."""


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

VALIDATOR_VIGNETTE_PROMPT = """You are a clinical validator checking whether a PTSD case vignette accurately reflects a specific cognitive model.

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
These causal connections must NOT appear — not explicitly, not implicitly.
{forbidden_edges}

What counts as an implicit violation:
- A and B described together in a way that implies one causes the other
- One component immediately follows another, implying causation
- The narrative "cycle" passes through a forbidden link
→ FAIL if any forbidden edge appears. When genuinely uncertain about a forbidden edge, FAIL.

STEP 4 — VERDICT
Before returning your verdict, confirm:
- You have NOT flagged any edge from the REQUIRED list as a violation.
- Every flagged violation comes from the FORBIDDEN list.

Return a JSON object with two fields:
- "verdict": exactly "PASS" or "FAIL"
- "reasoning": 1-2 sentences naming the specific edge(s) that caused a FAIL, or confirming all required edges are present for a PASS"""


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

