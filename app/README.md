# DepthCloud interface

The React, TypeScript, Vite, and Three.js interface for DepthCloud.

Use the root [README](../README.md) for setup and the capture workflow, and the [developer guide](../docs/DEVELOPMENT.md) for verification, architecture, and backend prerequisites.

From this directory:

```powershell
$env:npm_config_cache = Join-Path (Resolve-Path '..\backend').Path '.cache\npm'
npm ci
npm run dev
```

The development interface runs at [127.0.0.1:5173](http://127.0.0.1:5173/) and requires the backend at port 8765. `npm run build` generates `dist` for the backend to serve; `npm run lint` checks the frontend.
