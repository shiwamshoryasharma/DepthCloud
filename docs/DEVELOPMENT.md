# Development & architecture

[← Back to DepthCloud](../README.md)

## Environment preparation

The maintained environment is Windows x64 with embedded CPython 3.14.2 at `backend/python/python.exe`. Always invoke it through `backend/python.bat`. That launcher isolates user-site packages, uses project-local caches and temporary files, and enables offline Hugging Face loading.

A fresh clone does **not** include that runtime, pip/site configuration, CUDA wheels, model files, or npm dependencies. The original Python/CUDA bootstrap provenance is not recorded well enough to provide a tested clean-machine installer. Do not interpret the commands below as a complete bootstrap.

### Existing prepared runtime

From the repository root:

```powershell
Set-Location backend
.\python.bat --version
.\python.bat -m pip --version
.\python.bat -m pip check
```

The supported local runtime reports CPython 3.14.2. Its embedded `python314._pth` enables `import site`, and its installed packages must be visible through that runtime. System Python is not a drop-in replacement for this verified configuration.

To reinstall application dependencies **into an already prepared runtime with its working CUDA packages available**, run from `backend`:

```powershell
.\python.bat -m pip install setuptools-scm
.\python.bat -m pip install --no-build-isolation -r requirements.txt
.\python.bat -m pip check
```

| File | Purpose |
| :--- | :--- |
| [requirements.txt](../backend/requirements.txt) | Full resolved environment, including local vendored Depth Pro and pycolmap |
| [requirements.in](../backend/requirements.in) | Historical direct-dependency list; not a replacement for the full lock snapshot |
| [runtime-constraints.txt](../backend/runtime-constraints.txt) | Protects the prepared torch, torchvision, NumPy, and OpenCV versions |
| [python.bat](../backend/python.bat) | Runtime isolation and project-local cache settings |
| [.env.example](../backend/.env.example) | Public configuration example; existing private `.env` files must be preserved |

The pinned CUDA packages are `torch==2.14.0+cu132` and `torchvision==0.29.0+cu132`. Their original wheel/index source remains unverified; a normal package-index install on another machine may not resolve them. Establish trusted compatible runtime/wheel sources and validate them before claiming fresh-machine support. Do not silently downgrade the working numerical stack.

Depth Pro is vendored at revision `9e65e4dbe9568d23c546fcec53302b10445e109e`. Its [packaging adaptation](../backend/vendor/ml-depth-pro/DEPTHCLOUD_PATCH.md) relaxes upstream NumPy `<2` to `>=2,<3` and sets a local version. Local GPU inference passed previously; this is not an upstream compatibility guarantee. The native `pycolmap==4.2.0` Windows wheel performs camera reconstruction on CPU in this environment.

### Frontend development

Build and install using the [README commands](../README.md#2-build-the-interface). Keep the backend running in one terminal. In a second terminal, from the repository root:

```powershell
Set-Location app
npm run dev
```

Open [the Vite interface](http://127.0.0.1:5173/). Vite proxies `/api`, `/docs`, `/openapi.json`, and `/ws` to port 8765. Run `npm run build` after frontend edits to update the production interface served by the backend.

## Verification

From the repository root:

```powershell
Set-Location backend
.\python.bat -m pytest -q --basetemp .cache/pytest-publication
.\python.bat -m ruff check --no-cache src tests run.py run_mcp.py
.\python.bat -m pip check
Set-Location ..\app
npm run lint
npm run build
```

Verified again on **2026-09-25**: **100 backend tests passed** (21.94 seconds), Ruff and frontend lint passed, the production build succeeded, and `pip check` found no broken requirements. The suite includes isolated API capture storage, camera worker lifecycle, stream backpressure, reference matching, saved-set sealing/recovery, geometry validation, silhouette constraints, and occlusion handling. Hardware/model inference evidence is separate from automated tests.

Known warnings: Vite reports a large lazy Three.js renderer chunk, and previous browser runs emitted an upstream `THREE.Clock` deprecation warning. No fresh-machine installation, full 1,000-image hardware scan, full 360° physical accuracy test, or multi-hour soak is claimed.

## Source map

```text
DepthCloud/
├── app/                         React + TypeScript + Vite
│   └── src/
│       ├── components/          Camera, references, capture, 3D, calibration
│       ├── hooks/               Camera WebSocket lifecycle
│       └── services/            HTTP client
├── backend/
│   ├── src/depthcloud/
│   │   ├── api/                 Camera, reconstruction, system routes
│   │   ├── camera/              Isolated capture, buffers, calibration
│   │   ├── models/              Local Depth Pro and DETR ownership
│   │   ├── vision/              Appearance matching, masks, stability
│   │   ├── workflow/            Sessions, job ownership, collection
│   │   ├── reconstruction/      SfM, visual hull, legacy fusion, export
│   │   └── integrations/        Read-only MCP and Isaac metadata
│   ├── tests/                   Backend regression suite
│   └── vendor/ml-depth-pro/     Pinned third-party source and notices
└── docs/                        Public development documentation
```

### Ownership and reconstruction

OpenCV capture runs in a separate process because native camera drivers can block. A two-frame mailbox feeds a count/byte-bounded FIFO. Original frame arrays are immutable; JPEG preview work runs separately, and each WebSocket client can have one unacknowledged image.

`ReconstructionService` owns one worker/job at a time. Capture-set, continuous, and batch modes share that ownership and model inference locks. Cancellation occurs between operations; some native/model calls cannot be interrupted immediately. Use one backend process, without multiple Uvicorn workers.

The default saved-set path is:

1. `CaptureScanner` identifies steady frames using detection references; `CaptureDataset` writes original images and masks with an atomic manifest.
2. `sfm.py` uses sequential verified SIFT matching, triangulation, and bundle adjustment. One connected camera component is retained.
3. `multiview.py` refines masks and computes one metric scale from original-context depth estimates and triangulated scene points.
4. `visual_hull.py` intersects all registered silhouette constraints, samples boundaries at subpixel positions, extracts a surface, checks topology/projection agreement, and projects visible colors.
5. The mesh and integrity hash are persisted. The viewer loads binary buffers; OBJ mesh and PLY cloud exports use the first accepted camera coordinate frame.

Legacy depth fusion, continuous TSDF scanning, and selected-view batch registration are retained as explicit alternatives. They are not the primary saved-set geometry method.

### Budgets and persistence

| Resource | Current policy |
| :--- | :--- |
| Preview FIFO | Default 10 frames; maximum 128; default 128 MiB budget |
| Manual saved views | Default 24 views / 512 MiB |
| Capture set | Up to 1,000 images; 16 GiB disk budget; 1 GiB free reserve |
| Reconstruction masks | Up to 1,280-pixel width; aggregate 256 MiB mask budget |
| Surface | Up to 200,000 triangles after simplification |
| Capture storage | `backend/.cache/capture-session`; survives backend restart |
| Manual gallery / legacy results | RAM-only; lost when the backend exits |

These limits are not a total process-memory cap. Float distance fields, native geometry, GPU model state, IPC, and renderer buffers require additional memory.

Export coordinates are `(x, y, z)` in camera axes (+X right, +Y down, +Z forward). The Three.js display applies `(x, -y, -z)`; the Isaac descriptor maps to `(z, -x, -y)`. Apply the target conversion once. Orientation presets are relative to the reference camera, not semantic front/up recognition.

## API and integrations

Use [interactive OpenAPI](http://127.0.0.1:8765/docs) for exact payloads and validation. Useful read-only routes include:

| Route | Purpose |
| :--- | :--- |
| `GET /api/health` | Service health |
| `GET /api/camera/status` | Capture and buffer state |
| `GET /api/reconstruction/state` | Job, capture-set, and result state |
| `GET /api/reconstruction/capture/report` | Alignment and shape report |
| `GET /api/reconstruction/export/obj` | Current mesh |
| `GET /api/reconstruction/export/ply` | Current point cloud |
| `GET /api/integration/isaac` | Coordinate transform and export descriptor |

### MCP

With the backend already running, an MCP client can launch the stdio process from `backend`:

```powershell
.\python.bat run_mcp.py
```

Eight annotated read-only tools proxy the local HTTP API. They expose system/camera/job/session/geometry/Isaac metadata without starting another camera/model owner or offering shell execution. Client launch configuration depends on that client's Windows stdio support.

### Isaac Sim

The integration explicitly reports `not_connected`. It supplies export URLs, estimated dimensions, units, and coordinate transforms. A running simulator connection, scene import automation, and robot control are not implemented.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `python.exe` not found | The prepared runtime is missing from `backend/python`; Git does not distribute it |
| Missing Depth Pro / DETR files | Confirm the exact local model layout in the README |
| UI unavailable on port 8765 | Build `app/dist`, then restart the backend |
| Camera cannot open | Close competing camera apps; try the correct index and DirectShow/Media Foundation |
| Low measured FPS | Inspect actual capture telemetry; negotiated/requested FPS is not achieved FPS |
| Views rejected or shape distorted | Review masks, stationary-scene conditions, connected camera coverage, and calibration |
| Model/mesh loading is slow | First load differs from warmed execution; inspect CPU/GPU/RAM telemetry |
| Stop takes time | Cooperative cancellation waits for the current native/model operation |

## Continuation

The current local set has 41 images, 24 reference snapshots, and a persisted 52,114-triangle mesh. Only 19 images share the accepted camera model, spanning about 68°. Preserve the set. Next accuracy work needs overlapping captures connecting missing sides and an independent known measurement; do not merge disconnected components or weaken gates merely to increase counts.

Local development memory lives under the ignored `.agents/` directory on the maintained workstation. It is not part of a clone. Public setup, behavior, and limitations are documented here so external users do not need that private history.
