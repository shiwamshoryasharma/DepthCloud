import { useState } from 'react'
import { Eye, RefreshCw, Trash2 } from 'lucide-react'
import { command } from '../services/api'
import type { Observation, ReconstructionState } from '../types'

export function ViewGallery({ state, online, onRecapture, onError }: { state: ReconstructionState | null; online: boolean; onRecapture: (view: Observation) => void; onError: (message: string) => void }) {
  const [activeId, setActiveId] = useState<string | null>(null)
  const [kind, setKind] = useState('original')
  const [boxes, setBoxes] = useState(true)
  const [mask, setMask] = useState(false)
  const [pending, setPending] = useState(false)
  const views = state?.views || []
  const active = views.find(v => v.id === activeId)
  const locked = !online || state?.busy || pending
  async function update(view: Observation, method: 'PATCH' | 'DELETE') {
    setPending(true)
    try { await command('/reconstruction/views/' + view.id, method === 'PATCH' ? { selected: !view.selected } : undefined, method) }
    catch (e) { onError((e as Error).message) }
    finally { setPending(false) }
  }
  const available = (value: string) => !!active?.result.depth_stats || ['original', 'crop'].includes(value) || (!!active?.result.reference_only && ['mask', 'object'].includes(value))
  const mode = available(kind) ? kind : 'original'
  return <section className="panel gallery-panel">
    <div className="panel-heading"><div><span className="eyebrow">OBSERVATIONS</span><h2>Saved reference crops <span className="subtle-badge">{views.length} / {state?.max_views || 24}</span></h2></div><span className="mono small">{((state?.memory_bytes || 0) / 1024 ** 2).toFixed(1)} MB</span></div>
    {views.length === 0 ? <div className="gallery-empty">Capture the same object from several angles. Checked crops identify the object during image collection. Saved-view batch reconstruction remains available under advanced controls.</div> : <div className="view-gallery">{views.map((view, index) => <article key={view.id} className={'view-card ' + (activeId === view.id ? 'selected' : '')}>
      <button className="view-thumbnail" onClick={() => setActiveId(view.id)} aria-label={'Inspect ' + view.label}><img src={'/api/reconstruction/views/' + view.id + '/image/crop?v=' + view.sequence_id} alt={view.label + ' crop'} loading="lazy" /><span>{String(index + 1).padStart(2, '0')}</span></button>
      <label className="view-selection"><input type="checkbox" checked={view.selected} disabled={locked} onChange={() => void update(view, 'PATCH')} aria-label={'Include ' + view.label} /><b>{view.label}</b></label>
      <div className={'view-status ' + (view.error ? 'failed' : '')}>{view.status.replaceAll('_', ' ')}</div>
      <span className="mono view-time">#{view.sequence_id} · {new Date(view.timestamp * 1000).toLocaleTimeString()}</span>
      <div className="view-actions"><button aria-label={'Inspect details for ' + view.label} onClick={() => setActiveId(activeId === view.id ? null : view.id)}><Eye size={13} /></button><button disabled={locked} aria-label={'Recapture ' + view.label} onClick={() => onRecapture(view)}><RefreshCw size={13} /></button><button disabled={locked} aria-label={'Delete ' + view.label} onClick={() => void update(view, 'DELETE')}><Trash2 size={13} /></button></div>
      {view.error && <p className="inline-error">{view.error}</p>}
    </article>)}</div>}
    {active && <div className="view-inspector">
      <div className="inspection-heading"><h3>{active.label} · inspection</h3><button className="text-button" onClick={() => setActiveId(null)}>Close inspection</button></div>
      <div className="mode-tabs">{['original', 'crop', 'depth', 'mask', 'object'].map(value => <button key={value} aria-pressed={mode === value} disabled={!available(value)} onClick={() => setKind(value)}>{value === 'object' ? 'Object only' : value}</button>)}</div>
      <div className="inspection-image"><img src={'/api/reconstruction/views/' + active.id + '/image/' + mode + '?boxes=' + boxes + '&mask_overlay=' + mask + '&v=' + active.status} alt={active.label + ' ' + mode + ' inspection'} /></div>
      <div className="inspection-toggles"><label><input type="checkbox" checked={boxes} onChange={e => setBoxes(e.target.checked)} />Detection boxes</label><label><input type="checkbox" checked={mask} onChange={e => setMask(e.target.checked)} />Mask overlay / object depth</label></div>
      <p className="small muted">ROI: [{active.roi_pixels.join(', ')}] · {active.width} × {active.height} original pixels</p>
      {active.result.reference_only && <p className="small muted">Detection reference only. This saved image is not fused into the live reconstruction.</p>}
      {active.result.target && <p className="small">Target: <b>{active.result.target.label}</b> · {(active.result.target.score * 100).toFixed(1)}% detection confidence</p>}
      {active.result.warning && <p className="warning-text small">{active.result.warning}</p>}
      {active.result.depth_stats && <><dl className="result-metrics"><div><dt>Object depth</dt><dd>{active.result.depth_stats.minimum.toFixed(3)}–{active.result.depth_stats.maximum.toFixed(3)} m</dd></div><div><dt>Valid depth</dt><dd>{active.result.depth_stats.valid_percent.toFixed(1)}%</dd></div><div><dt>Focal length</dt><dd>{active.result.focal_px?.toFixed(1)} px</dd></div><div><dt>Intrinsics</dt><dd>{active.result.intrinsics?.source.replaceAll('_', ' ')}</dd></div></dl><a className="download-link" href={'/api/reconstruction/views/' + active.id + '/depth.npy'}>Download raw depth (.npy, meters)</a></>}
      {!!active.result.depth_outliers_removed && <p className="small muted">Depth cleanup excluded {active.result.depth_outliers_removed.toLocaleString()} foreground pixels; raw depth is unchanged.</p>}
      {active.result.registration && <p className="small muted">Registration: {active.result.registration.method} · fitness {active.result.registration.fitness.toFixed(3)} · overlap {active.result.registration.overlap.toFixed(3)} · RMSE {active.result.registration.rmse.toFixed(4)} m</p>}
    </div>}
    <p className="small muted session-note">Check every angle to use for object identification. Captured references survive camera stop. Starting collection saves a copy of included references with the temporary capture set.</p>
  </section>
}

