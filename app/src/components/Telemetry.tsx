import { Activity, Cpu, HardDrive, Timer } from 'lucide-react'
import type { CameraStatus, SystemStatus } from '../types'

export function Telemetry({ camera, system, history, online }: { camera: CameraStatus | null; system: SystemStatus | null; history: number[]; online: boolean }) {
  const value = (n: number | undefined, digits = 1) => online && n !== undefined ? n.toFixed(digits) : '—'
  const points = history.map((n, i) => (i / Math.max(1, history.length - 1) * 240) + ',' + (64 - Math.min(120, n) / Math.max(30, ...history) * 52)).join(' ')
  return <aside className="inspector">
    <section className="panel telemetry-panel">
      <div className="section-title"><span className="eyebrow">LIVE DIAGNOSTICS</span><Activity size={16} /></div>
      <div className="main-metric"><strong>{value(camera?.capture_fps)}</strong><span>capture fps</span></div>
      <svg className="fps-chart" viewBox="0 0 240 72" role="img" aria-label="Measured capture frame rate over the last 30 seconds">
        <path d="M0 16H240 M0 40H240 M0 64H240" className="chart-grid" />
        {history.length > 1 && <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2" />}
      </svg>
      <div className="chart-labels"><span>LAST 30 SECONDS</span><span>NOW</span></div>
      <dl className="metrics">
        <div><dt>Preview rate</dt><dd>{value(camera?.processing_fps)} <span>fps</span></dd></div>
        <div><dt>JPEG processing</dt><dd>{value(camera?.processing_latency_ms)} <span>ms</span></dd></div>
        <div><dt>Capture → preview</dt><dd>{value(camera?.preview_latency_ms)} <span>ms</span></dd></div>
        <div><dt>Frames received</dt><dd>{value(camera?.frames_received, 0)}</dd></div>
        <div><dt title="Capture mailbox drops plus frames skipped by preview encoding">Skipped processing</dt><dd>{value(camera?.frames_dropped, 0)}</dd></div>
        <div><dt title="Normal FIFO evictions; counted separately from processing drops">FIFO evictions</dt><dd>{value(camera?.buffer_evictions, 0)}</dd></div>
      </dl>
    </section>
    <section className="panel memory-panel">
      <div className="section-title"><span className="eyebrow">MEMORY BUDGET</span><HardDrive size={15} /></div>
      <div className="memory-value">{value(camera ? camera.buffer_bytes / 1024 ** 2 : undefined)} <span>/ {value(camera ? camera.buffer_memory_limit / 1024 ** 2 : undefined, 0)} MB</span></div>
      <div className="progress-track"><div style={{ width: camera ? Math.min(100, camera.buffer_bytes / camera.buffer_memory_limit * 100) + '%' : '0%' }} /></div>
      <p className="small muted">Frame buffer · bounded in RAM</p>
      <div className="memory-bottom"><span>Backend process</span><span className="mono">{value(system?.memory_mb, 0)} MB</span></div>
    </section>
    <section className="panel device-panel">
      <div className="section-title"><span className="eyebrow">COMPUTE & MODELS</span><Cpu size={16} /></div>
      <h3>{online ? system?.device.name || 'Detecting device…' : 'Device disconnected'}</h3>
      <span className="device-tag">{online ? system?.device.type.toUpperCase() || '—' : '—'}</span>
      <div className="model-row"><span>Depth Pro</span><span className={system?.models.depth_pro?.files_present && online ? 'model-ok' : 'muted'}>{online ? system?.models.depth_pro?.loading ? 'Loading…' : system?.models.depth_pro?.loaded ? 'Loaded' : system?.models.depth_pro?.files_present ? 'Ready locally' : 'Missing files' : '—'}</span></div>
      <div className="model-row"><span>DETR ResNet-50</span><span className={system?.models.detr?.files_present && online ? 'model-ok' : 'muted'}>{online ? system?.models.detr?.loading ? 'Loading…' : system?.models.detr?.loaded ? 'Loaded' : system?.models.detr?.files_present ? 'Ready locally' : 'Missing files' : '—'}</span></div>
      <p className="small muted">VRAM allocated / reserved: {value(system?.vram_allocated_mb, 0)} / {value(system?.vram_reserved_mb, 0)} MB<br />Depth latency: {value(system?.models.depth_pro?.latency_ms ?? undefined, 0)} ms · DETR: {value(system?.models.detr?.latency_ms ?? undefined, 0)} ms</p>
      <div className="uptime"><Timer size={13} />{online && system ? Math.floor(system.uptime_seconds / 60) + 'm ' + Math.floor(system.uptime_seconds % 60) + 's uptime' : 'Waiting for backend'}</div>
    </section>
  </aside>
}

