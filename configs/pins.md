# Dependency Pin Exception Log

This document records every deviation from the default policy of using the latest stable release for project dependencies. Update it whenever a component is locked to a non-current version.

## How to Use
- Add a new entry for each dependency pinned below the newest stable release.
- Include a short rationale, the impacted components, mitigation plan, and a review date.
- Remove an entry once the pin is lifted and the dependency is upgraded.

## Current Pins

| Dependency | Version | Latest Stable | Reason for Pin | Impacted Areas | Mitigation / Exit Criteria | Review Date | Owner |
|------------|---------|---------------|----------------|----------------|----------------------------|-------------|-------|
| _None_     | -       | -             | -              | -              | -                          | -           | -     |

## Review Checklist
- Confirm no new upstream release resolves the blocking issue.
- Validate upgrade in staging environment when feasible.
- Update automated tests or compatibility layers if required.
- Record decision outcomes (retain pin or upgrade) with timestamp and reviewer initials.
