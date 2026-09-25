# Local Python 3.14 packaging adaptation
Source: https://github.com/apple/ml-depth-pro
Pinned upstream commit: 9e65e4dbe9568d23c546fcec53302b10445e109e

Only pyproject.toml changed:
- version 0.1 -> 0.1+depthcloud314 to distinguish the local build
- numpy<2 -> numpy>=2,<3 because CPython 3.14 uses NumPy 2 and the application already has NumPy 2.4.1

Inference implementation and model weights are unchanged. This is a local packaging adaptation, not an upstream compatibility guarantee. Import/dependency checks and actual local CUDA checkpoint loading/inference passed on 2026-09-23. See the [DepthCloud developer guide](../../../docs/DEVELOPMENT.md#verification) for scope and accuracy limits. Preserve the upstream license files.
