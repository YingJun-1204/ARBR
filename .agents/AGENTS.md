# Workspace Rules for Git Operations

## Iron Rule: Safe Non-Destructive Git Execution
When executing Git operations (e.g. `git push`, `git add`, `git commit`):
1. **Allowed 3-Step Flow ONLY**:
   - `git add .` (or `git add -A`)
   - `git commit -m "<message>"`
   - `git push`
2. **STRICTLY FORBIDDEN Destructive Commands**:
   - NEVER run `git restore` or `git restore .`
   - NEVER run `git checkout -- .`
   - NEVER run `git reset --hard`
   - NEVER switch branches with `git checkout <branch>` without committing first
3. **Respect User Local State**:
   - Whatever files the user deleted, added, or edited in their working directory MUST be respected. Never run any command that resets or reverts local changes.
