# Fieldlink - Cross-Repository Connections

Fieldlink provides direct references to companion repositories in the JinnZ2 ecosystem.
These links establish context bridges so that AI agents and human collaborators can
navigate between related projects without losing context.

## Linked Repositories

| Repository | Purpose | Link |
|------------|---------|------|
| **Nexus-emergency-management** | Emergency management interface and resources | [GitHub](https://github.com/JinnZ2/Nexus-emergency-management) |
| **Infrastructure-assistance** | Resilience resources for decaying infrastructure | [GitHub](https://github.com/JinnZ2/Infrastructure-assistance) |

## How Fieldlinks Work

Each `.fieldlink.json` file in this directory describes a connection to an external
repository, including:

- **What it links to** (repo URL, branch, key paths)
- **Why the link exists** (shared concepts, data flow, dependency)
- **Where to start** (entry points for exploration)

These files are machine-readable so that AI agents can automatically discover and
traverse related projects.

## Usage

```python
# Example: Load fieldlinks programmatically
import json
from pathlib import Path

fieldlinks = {}
for f in Path("fieldlink").glob("*.fieldlink.json"):
    with open(f) as fh:
        fieldlinks[f.stem] = json.load(fh)
```
