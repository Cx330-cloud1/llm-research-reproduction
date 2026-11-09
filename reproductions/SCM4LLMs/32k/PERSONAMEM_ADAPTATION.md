# SCM4LLMs -> PersonaMem 32k Adaptation

## Scope

This branch adapts the original SCM4LLMs memory framework to the
PersonaMem 32k benchmark.

This is NOT an original-paper SCM experiment.

Original SCM baseline commit:

26216044f5d4caccc0303d86667b491ead11d878

Security/compatibility parent branch:

modern-api-adapter

PersonaMem adaptation branch:

personamem-adapter

---

# 1. PersonaMem dataset used

Files:

- questions_32k.csv
- shared_contexts_32k.jsonl

Local location:

/root/autodl-tmp/paper_repro/shared_data/personamem_32k/

Observed dataset statistics:

- questions: 589
- shared contexts: 37
- missing shared_context_id mappings: 0
- invalid end_index_in_shared_context values: 0

Question type counts:

- track_full_preference_evolution: 139
- recall_user_shared_facts: 129
- recalling_the_reasons_behind_previous_updates: 99
- suggest_new_ideas: 93
- generalizing_to_new_scenarios: 57
- provide_preference_aligned_recommendations: 55
- recalling_facts_mentioned_by_the_user: 17

---

# 2. Hard no-leakage rule

For every question, the only allowed historical context is:

visible_context =
    full_context[:end_index_in_shared_context]

No message at index >= end_index_in_shared_context may be used for:

- SCM Turn construction;
- memory embedding;
- summarization;
- retrieval;
- controller decisions;
- generation;
- token-budget calculations;
- any preprocessing whose output is exposed to the model.

The context MUST be truncated before any SCM-specific processing.

This is a hard benchmark constraint.

---

# 3. Observed PersonaMem message structure

Roles observed:

- system
- user
- assistant

No other roles were observed.

Across the 37 complete shared contexts:

- system messages: 195
- user messages: 3363
- assistant messages: 2904

Every shared context starts with:

system -> user -> assistant -> user -> assistant

However, the full histories are not strictly alternating.

Observed same-role adjacency:

- user -> user: 446
- assistant -> assistant: 11

System messages may also appear inside a conversation.

---

# 4. Persona system-message behavior

Every shared context contains multiple system messages.

Observed:

- all 37 contexts contain repeated system persona messages;
- within each individual context, every system message is textually
  identical;
- contexts with multiple unique system-message contents: 0.

Therefore the non-initial system messages are repeated persona
reinjections, not observed persona updates.

IMPORTANT:

The adaptation must not silently remove repeated system messages merely
because their contents are identical.

Removing them would alter:

- chronological structure;
- token distance;
- context budget;
- persona recency.

How system reinjections are represented inside SCM remains an explicit
adapter design decision and must be documented.

---

# 5. Boundary behavior

Across the 589 question-specific visible contexts:

- visible context ends in user: 400 questions
- visible context ends in assistant: 189 questions

Therefore PersonaMem frequently places the question boundary after a
user message but before its following assistant response.

Example observed boundary:

index 179: user       visible
index 180: assistant  visible
index 181: user       visible
index 182: assistant  hidden

For this case, index 181 MUST be preserved while index 182 MUST remain
completely inaccessible.

The adapter MUST NOT:

- drop the final visible user message;
- read the hidden assistant response;
- generate a synthetic historical assistant response to complete the turn.

---

# 6. Consecutive same-role messages

PersonaMem contains consecutive messages with the same role.

Observed:

- user -> user: 446
- assistant -> assistant: 11

After temporarily excluding system messages and losslessly collapsing
consecutive messages with the same user/assistant role, no same-role
adjacency remains.

Therefore a lossless role-run representation is possible.

If role-run collapsing is used, all original message contents and their
order must be preserved.

No text may be summarized or discarded during structural conversion.

---

# 7. Original SCM structural mismatch

Original SCM memory units are based on:

Turn(
    user_input,
    system_response,
    user_sys_text,
    summ,
    embedding
)

PersonaMem is a general message stream and can contain:

- repeated system persona injections;
- multiple consecutive user messages;
- multiple consecutive assistant messages;
- a final visible unpaired user message.

Therefore PersonaMem cannot be mapped by simply grouping every two
messages into one SCM Turn.

Any mapping rule is an adaptation and must be explicitly reported.

---

# 8. Prohibited adaptation shortcuts

The implementation must never:

1. process the complete shared context before applying end_index;
2. read hidden assistant responses;
3. fabricate historical assistant messages;
4. silently delete visible user messages;
5. silently remove repeated persona reinjections;
6. reorder historical messages;
7. regenerate existing historical assistant responses;
8. call a model to rewrite history before the no-leakage boundary has
   been applied;
9. treat the PersonaMem adaptation as an original SCM-paper experiment.

---

# 9. Historical response policy

Existing PersonaMem user/assistant history must be reused exactly as
provided.

Historical assistant responses must NOT be regenerated.

Only SCM-specific derived artifacts may later be generated when
necessary, such as:

- summaries;
- embeddings;
- controller decisions.

All such derived artifacts must be based only on visible history.

---

# 10. Evaluation target

PersonaMem questions contain:

- question_id
- question_type
- user_question_or_message
- correct_answer
- all_options
- shared_context_id
- end_index_in_shared_context

The final PersonaMem evaluation must preserve the official question and
answer choices.

The historical memory mechanism and final answer-generation model must
be separated conceptually:

SCM memory construction/retrieval
        ->
selected memory context
        ->
shared final answer-generation protocol
        ->
answer parser
        ->
accuracy

Published SCM-paper results must not be mixed with PersonaMem results.

---

# 11. Open adapter design decisions

The following decisions are intentionally NOT finalized yet:

1. how repeated system persona reinjections map into SCM memory;
2. how a visible history ending in an unpaired user message maps into
   SCM's previous-turn mechanism;
3. whether consecutive same-role messages are collapsed into role-runs
   before Turn construction;
4. how SCM summaries are produced for PersonaMem history;
5. which embedding model is used for the PersonaMem adaptation;
6. how the confirmed SCM non-terminating controller bug is guarded;
7. the unified final generation model and answer parser.

These decisions must be resolved explicitly before running benchmark
experiments.


---

# 12. Resolved structural normalization

The benchmark-facing loader and the SCM-specific adapter are separated.

The benchmark loader MUST first produce a lossless representation of the
question-visible PersonaMem history. It must not directly construct SCM
Turn objects.

Processing order:

1. load the full shared context;
2. apply:

       visible_messages =
           full_context[:end_index_in_shared_context]

3. preserve every visible message in its original order;
4. preserve every visible system message and its original position;
5. for user/assistant messages only, collapse consecutive messages with
   the same role into a lossless role-run;
6. pair completed:

       user-run -> assistant-run

   blocks as completed historical interactions;
7. if the visible history ends in a user-run, preserve that run
   separately as:

       pending_user_run

   and DO NOT construct a completed SCM Turn from it.

No assistant response may be fabricated for pending_user_run.

## Evidence for pending_user_run rule

Among the 589 question-visible histories:

- 400 end in a user message;
- 189 end in an assistant message.

For the 400 histories ending in user, the raw predecessor patterns were:

- 343: user -> assistant -> user
- 57: assistant -> user -> user

After losslessly collapsing consecutive same-role user/assistant
messages, the user/assistant role-runs alternate without same-role
adjacency.

Therefore the 57 user -> user cases do not require a separate special
case: they form one final user-run.

## System-message handling

Every question-visible history contains between 4 and 7 visible system
persona messages.

Within each shared context, all repeated system messages are textually
identical, but their repeated positions are part of the official
PersonaMem context.

The benchmark loader MUST preserve all of these system events.

How those system events are exposed to SCM retrieval/generation is NOT
yet resolved and belongs to the SCM-specific adaptation layer.

The benchmark loader must not silently deduplicate them.

