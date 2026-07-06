---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes
---

## Overview
Random fixes waste time and create new bugs. Quick patches mask underlying issues.

**Core principle:** ALWAYS find root cause before attempting fixes. Symptom fixes are failure.

## The Iron Law
```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

If you haven't completed Phase 1, you cannot propose fixes.

## When to Use
Use for ANY technical issue:
- Test failures
- Bugs in production
- Unexpected behavior
- Performance problems
- Build failures
- Integration issues

## The Four Phases
You MUST complete each phase before proceeding to the next.

### Phase 1: Root Cause Investigation
**BEFORE attempting ANY fix:**

1. **Read Error Messages Carefully**
   - Don't skip past errors or warnings
   - They often contain the exact solution
   - Read stack traces completely
   - Note line numbers, file paths, error codes

2. **Reproduce Consistently**
   - Can you trigger it reliably?
   - What are the exact steps?

3. **Check Recent Changes**
   - What changed that could cause this?
   - Git diff, recent commits

4. **Gather Evidence in Multi-Component Systems**
   - Log what data enters/exits components
   - Check state at each layer

### Phase 2: Pattern Analysis
**Find the pattern before fixing:**

1. **Find Working Examples**
   - Locate similar working code in same codebase

2. **Compare Against References**
   - Read reference implementation COMPLETELY

3. **Identify Differences**
   - List every difference, however small

### Phase 3: Hypothesis and Testing
**Scientific method:**

1. **Form Single Hypothesis**
   - State clearly: "I think X is the root cause because Y"

2. **Test Minimally**
   - Make the SMALLEST possible change to test hypothesis
   - One variable at a time

3. **Verify Before Continuing**
   - Did it work? Yes → Phase 4
   - Didn't work? Form NEW hypothesis

### Phase 4: Implementation
**Fix the root cause, not the symptom:**

1. **Create Failing Test Case**
   - Simplest possible reproduction
   - MUST have before fixing
   - Use the `superpowers:test-driven-development` skill

2. **Implement Single Fix**
   - Address the root cause identified
   - ONE change at a time

3. **Verify Fix**
   - Test passes now?
   - No other tests broken?
   - Issue actually resolved?

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see if it works"
- "Add multiple changes, run tests"
- "Skip the test, I'll manually verify"
- "It's probably X, let me fix that"

**ALL of these mean: STOP. Return to Phase 1.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Root Cause** | Read errors, reproduce, check changes | Understand WHAT and WHY |
| **2. Pattern** | Find working examples, compare | Identify differences |
| **3. Hypothesis** | Form theory, test minimally | Confirmed or new hypothesis |
| **4. Implementation** | Create test, fix, verify | Bug resolved, tests pass |
