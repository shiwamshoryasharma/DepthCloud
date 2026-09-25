import { lazy, Suspense, useState } from 'react'
import { AlertCircle, ArrowUpRight, Box, Crop, X } from 'lucide-react'
import { BufferPanel } from './components/BufferPanel'
import { CameraControls } from './components/CameraControls'
import { LivePreview } from './components/LivePreview'
import { Telemetry } from './components/Telemetry'
import { PreviewControls } from './components/PreviewControls'
import { RoiEditor } from './components/RoiEditor'
import { ViewGallery } from './components/ViewGallery'
import { LiveScanPanel } from './components/LiveScanPanel'
import { CaptureScanPanel } from './components/CaptureScanPanel'
import { PipelinePanel } from './components/PipelinePanel'
import { CalibrationPanel } from './components/CalibrationPanel'
import { useCameraStream } from './hooks/useCameraStream'
import { command } from './services/api'
import type { FrozenFrame, Observation, ROI } from './types'
import './App.css'
import './studio.css'
import './compact.css'

const WorldViewer = lazy(() => import('./components/WorldViewer').then(module => ({ default: module.WorldViewer })))
function App() {
  const stream = useCameraStream()
  const [error, setError] = useState('')
  const [setupOpen, setSetupOpen] = useState(false)
  const [freezing, setFreezing] = useState(false)
  const [frozen, setFrozen] = useState<FrozenFrame | null>(null)
  const [replacement, setReplacement] = useState<Observation | null>(null)
  const online = stream.connection === 'online'
  const running = online && stream.camera?.state === 'running'
  async function freeze(view: Observation | null = null) {
    setFreezing(true); setError('')
    try { const frame = await command<FrozenFrame>('/reconstruction/freeze'); setReplacement(view); setFrozen(frame) }
    catch (e) { setError((e as Error).message) }
    finally { setFreezing(false) }
  }
  async function capture(roi: ROI, label: string) {
    if (!frozen) return
    await command('/reconstruction/views', { snapshot_id: frozen.id, roi, label, replace_id: replacement?.id || null })
    stream.record((replacement ? 'Recaptured ' : 'Captured ') + label)
    setFrozen(null); setReplacement(null)
  }
  return <div className="app-shell studio-shell compact-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="DepthCloud home"><span className="brand-mark"><Box size={20} /></span>DepthCloud</a><nav aria-label="Workspace sections"><a href="#setup" onClick={() => setSetupOpen(true)}>1 · Setup</a><a href="#capture-set">2 · Capture</a><a href="#reconstruction">3 · Mesh</a></nav><div className="topbar-right"><span className="connection"><span className={'status-dot ' + (online ? 'green' : 'amber')} />{online ? 'Connected' : 'Backend offline'}</span><a className="docs-link" href="/docs" target="_blank" rel="noreferrer">API <ArrowUpRight size={13} /></a></div></header>
    <main>
      <div className="workspace-heading"><div><h1>Object reconstruction</h1><p>Set references. Collect steady views. Reconstruct the saved set.</p></div></div>
      {(error || stream.camera?.error) && <div className="error-banner" role="alert"><AlertCircle size={18} /><span>{error || stream.camera?.error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
      <details className="panel setup-drawer" id="setup" open={setupOpen} onToggle={event => setSetupOpen(event.currentTarget.open)}><summary>Camera setup <span>{running ? 'Camera running' : 'Start here · camera is off'}</span></summary><div className="setup-grid"><CameraControls camera={stream.camera} online={online} onError={setError} onRecord={stream.record} /><BufferPanel camera={stream.camera} online={online} onError={setError} /><CalibrationPanel state={stream.reconstruction} online={online} onError={setError} /></div></details>
      <CaptureScanPanel state={stream.reconstruction} running={running} online={online} onError={setError} />
      <div className="studio-grid">
        <div className="acquisition-column"><section className="acquisition-group"><LivePreview camera={stream.camera} image={stream.image} online={online} onLoad={stream.acknowledge} /><div className="capture-action"><button className="button primary" disabled={!running || freezing || stream.reconstruction?.busy} onClick={() => void freeze()}><Crop size={16} />{freezing ? 'Freezing…' : 'Add detection reference'}</button><span>Crop the same object from different angles.</span></div><details className="preview-drawer"><summary>Preview appearance</summary><PreviewControls current={stream.camera?.preview_options} online={online} onError={setError} /></details></section><details className="panel reference-drawer" open><summary>Detection references <span>{stream.reconstruction?.views.filter(view => view.selected).length || 0} included</span></summary><ViewGallery state={stream.reconstruction} online={online} onRecapture={view => void freeze(view)} onError={setError} /></details></div>
        <div className="reconstruction-column" id="reconstruction"><Suspense fallback={<section className="panel loading-world">Loading 3D workspace…</section>}><WorldViewer result={stream.reconstruction?.result || null} /></Suspense><p className="mesh-note">The mesh appears after collection and reconstruction. Cameras are aligned together using the stationary scene. Object outlines constrain one common surface. Additional views improve coverage; unseen detail remains estimated.</p></div>
      </div>
      <details className="panel advanced-drawer"><summary>Advanced tools & diagnostics</summary><div className="advanced-grid"><LiveScanPanel state={stream.reconstruction} running={running} online={online} onError={setError} /><PipelinePanel state={stream.reconstruction} online={online} onError={setError} /></div><Telemetry camera={stream.camera} system={stream.system} history={stream.fpsHistory} online={online} /><div className="activity-list">{stream.activity.slice(0, 5).map(event => <div className="event" key={event.id}><span>{event.message}</span><time>{event.time}</time></div>)}</div></details>
    </main>
    <footer><span>Local models · project-local temporary storage</span><span>Estimated geometry · meters</span></footer>
    {frozen && <RoiEditor key={frozen.id} frame={frozen} initialLabel={replacement?.label || 'Reference ' + ((stream.reconstruction?.views.length || 0) + 1)} onClose={() => { setFrozen(null); setReplacement(null) }} onCapture={capture} />}
  </div>
}
export default App
