---
name: prepare-pr
description: "Prepare an E3-SSE pull request by validating the repository, summarizing scientific and code changes, documenting tests, risks, provenance impact, and follow-up work."
---

# Prepare Pull Request

Before PR preparation:

1. inspect the diff;
2. run `.agents/scripts/validate_repo.sh`;
3. confirm raw data are not tracked;
4. identify affected scientific claim/workflow;
5. identify tests performed;
6. identify unresolved limitations;
7. identify configuration/schema changes;
8. identify whether existing results need regeneration.

The PR summary should contain:

- purpose;
- implementation;
- scientific implications;
- validation;
- data/provenance impact;
- risks;
- follow-up work.
