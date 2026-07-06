---
name: windows-demo-environment
description: Use when working in this Windows demo project, especially before running commands, Python training or tuning scripts, editing Chinese comments/Markdown/log files, reading Chinese paths, or handling encoding, conda, PowerShell, CUDA, or DataLoader settings.
---

# Windows Demo Environment

## Overview

This project runs on Windows and may contain Chinese text in comments, Markdown files, folder names, loss logs, terminal output, and JSON-adjacent experiment artifacts. Treat environment setup and encoding as part of correctness, not a cosmetic detail.

## Command Rules

- In this project folder, activate the `demo` conda environment before running commands.
- Prefer this robust Windows pattern for Python commands:

```powershell
cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python <script-or-module>"
```

- Use PowerShell `-LiteralPath` for paths, especially paths that may contain `+`, spaces, or Chinese characters.
- Do not rely on `activate.ps1`; it may be unavailable. Prefer `activate.bat` through `cmd /d /c`.
- Do not run multiple conda activations in parallel if they could share temporary activation state.

## Git / GitHub Sync Rules

- Do not automatically push to GitHub or sync local repository changes unless the user explicitly requested it in their prompt.
- If you are unsure whether to sync/push to GitHub, ask the user for permission and only proceed after getting a positive confirmation. Otherwise, do not run `git push`.

## Encoding Rules

- Assume non-code files can contain Chinese: `.md`, `.txt`, `.log`, `.json`, experiment summaries, and folders such as `loss日志` or `md文件`.
- Preserve UTF-8 when editing files. When Python reads logs or generated outputs, use `encoding="utf-8", errors="replace"` or `errors="ignore"` only when loss is acceptable.
- When forwarding subprocess output to the console on Windows, sanitize output for the current console encoding to avoid GBK/Unicode crashes.
- Do not "fix" Chinese text by transliterating or deleting it. Preserve the user's Chinese unless explicitly asked to translate.

## Training And Tuning Rules

- Always keep DataLoader workers at zero on Windows: pass `--num_workers 0` and keep configs/scripts at `num_workers=0`.
- For long-running training or tuning commands, include the conda activation and UTF-8 code page setup in the command shown to the user.
- If a command emits Chinese logs or paths, prefer `chcp 65001` and Python UTF-8-safe output handling.

## Quick Checks

Before finalizing environment-sensitive work, verify:

- The command activates `demo`.
- The command or script uses `--num_workers 0`.
- Chinese paths/text are quoted and preserved.
- Logs are read/written with UTF-8-safe handling.
