# ADI Finite-Difference Methods for American Options under the Heston Model

Math/CS 714 final project, UW–Madison, Fall 2026 · Rohan Gupta, Ganesh Arivoli · Instructor: Yue Sun

We price American put options under the Heston stochastic-volatility model by solving the pricing PDE
with ADI (alternating direction implicit) finite-difference schemes: Douglas, Craig–Sneyd,
Modified Craig–Sneyd and Hundsdorfer–Verwer. The early-exercise constraint is handled with
Ikonen–Toivanen operator splitting. We study when the schemes stay accurate and stable: under strong
correlation, when the variance can reach zero, and near the exercise boundary. We then speed up the
solver with C++ multithreading and CUDA, and compare their performance.

Proposal: [`docs/proposal.pdf`](docs/proposal.pdf)

## Status

Phase 0 (setup). There is no solver code yet. Setup steps and the command to reproduce each figure
will be added here as code lands.
