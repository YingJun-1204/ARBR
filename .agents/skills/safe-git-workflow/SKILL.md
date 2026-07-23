---
name: safe-git-workflow
description: Strict safety rules for Git operations. MUST trigger whenever the user requests git actions, git upload, git push, or whenever executing git commands. Enforces standard non-destructive workflow and strictly forbids restore, reset, or checkout destructive commands.
---

# Safe Git Workflow

## Standard Safe Git Execution Flow
When performing git operations or when the user requests git upload / commit / push:
1. `git add .` (or `git add -A`) - Stage all current project modifications.
2. `git commit -m "<message>"` - Save local version snapshot.
3. `git push origin main` (or specified remote/branch) - Push committed version to remote repository.

Only these 3 standard commands are allowed for git upload. They are 100% safe and read/stage current local state without altering local files.

## Strictly Forbidden Destructive Git Commands
The agent is **STRICTLY FORBIDDEN** from running any of the following commands on behalf of the user unless the user explicitly commands it:
- `git restore` / `git restore .` (FORBIDDEN: reverts local changes or restores manually deleted files)
- `git checkout -- .` (FORBIDDEN: discards uncommitted code changes)
- `git reset --hard` (FORBIDDEN: wipes local working tree and resets to past commit)
- `git checkout <branch>` (FORBIDDEN: switches branch without committing, causing local file overwrites)

## Critical Guidelines
- Never run `git restore` to "fix" or clean `git status` output.
- Whatever files the user deleted or modified in their workspace, respect their local state exactly.
- Simply run `git add .` -> `git commit -m "..."` -> `git push`.
