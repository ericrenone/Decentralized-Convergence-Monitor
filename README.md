# Hyperbolic-Petri-Net-Simulator
Visual exploration of Petri net executions in the Poincaré disk


## Features

- Greedy forward simulation (fires first enabled transition)
- Efficient: hyperbolic distance & admissibility checked only periodically
- Deterministic 128-bit structural fingerprint (safe for negative weights)
- Natural termination detection
- Clean console progress with ASCII marking bars
- Export: `.txt` log + `.json` summary + `.png` trajectory plot
- Four plot styles: `classic` • `detailed` • `compact` • `bars`
- Single-file, minimal dependencies (matplotlib optional)
