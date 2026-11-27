# Issue template (for `local-agent` issues)

A `local-agent` issue is done by a small local model. Write it so that someone who has never seen the repo, and who will not make judgment calls, can finish it by following the steps. If you cannot write it that way, label it `claude` instead.

Copy this into the Linear issue description:

```markdown
## Goal
One or two sentences: what will be true when this is done, and why it matters.

## Before you start
- [ ] Things that must already be true (a merged issue, an installed library, a file that exists).
      If any is false, stop and report.

## Steps
1. Exact actions, in order. Name every file. Give exact commands.
2. When a step creates code, say what it is called, where it goes, and what it does.
   Give a code skeleton for anything longer than about 20 lines.
3. Say what NOT to touch if there is an obvious temptation.

## Acceptance
Commands that must pass, with what "pass" looks like:
    cd ng && cargo test -p <crate>      → all tests pass
    <command>                           → prints <expected>

## Out of scope
Things that look related but must not be done here.

+++ Agent notes
Branch name, the relevant spec section, gotchas, links.
+++
```

Rules:

- **One issue = one PR.** If the steps span two unrelated areas, split the issue.
- **Every number is given.** Tolerances, iteration limits, grid sizes, file paths: never "a suitable value".
- **Acceptance is a command, not a judgment.** "Looks right" is not acceptance.
- **Plain language.** No session jargon. Define any term the spec does not define.
- Dependencies use Linear's "blocked by" relation, not a label or a sentence.
