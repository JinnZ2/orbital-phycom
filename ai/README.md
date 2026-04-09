# AI - Safe Entry Point

Welcome. This folder is the designated entry point for AI agents working with
the Orbital PHYCOM codebase. It provides clear navigation, safe exploration
boundaries, and training resources.

## Quick Orientation

```
ai/
  README.md              <-- You are here. Start here.
  navigation.json        <-- Machine-readable map of the entire codebase
  safety_boundaries.md   <-- What's safe to read, run, and modify
  training/              <-- AI training resources & solution space explorer
    README.md            <-- Training overview
    explore_solution_space.py  <-- Interactive solution space exploration tool
    codebase_map.py      <-- Programmatic codebase discovery
    exercises.json       <-- Guided learning exercises
```

## For AI Agents

1. **Read `safety_boundaries.md`** first to understand safe operating limits
2. **Load `navigation.json`** to get a structured map of every module
3. **Run training exercises** in `training/` to build codebase familiarity

## For Human Collaborators

This folder helps AI assistants onboard faster and work more safely within the
project. The navigation index and safety boundaries reduce the risk of
unintended modifications and help the AI understand project context without
extensive back-and-forth.

## Core Principles

- **Safe by default**: AI agents start with read-only orientation
- **Progressive disclosure**: Simple overview first, deep detail on demand
- **Machine-readable**: JSON indexes for programmatic navigation
- **Human-reviewable**: All AI actions should be transparent and auditable
