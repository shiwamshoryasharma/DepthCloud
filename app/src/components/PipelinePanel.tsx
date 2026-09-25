import { useState } from 'react'
import { Download, Scan, Square, Trash2 } from 'lucide-react'
import { command } from '../services/api'
import type { PipelineOptions, ReconstructionState } from '../types'

const initial: PipelineOptions = { depth_cleanup: true, depth_outlier_strength: 8, threshold: .5, require_detection: false, depth_min: .05, depth_max: 10, stride: 2, voxel_size: .005, icp_distance: .03, icp_iterations: 60, min_fitness: .35, normal_radius: .02, outlier_neighbors: 20, outlier_std: 2, create_mesh: true, mesh_radius_factor: 2.5 }
export function PipelinePanel({ state, online, onError }: { state: ReconstructionState | null; online: boolean; onError: (s: string) => void }) {
  const [options, setOptions] = useState(initial)
  const [pending, setPending] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const busy = !!state?.busy || pending
  const selected = state?.views.filter(v => v.selected).length || 0
  async function action(name: string) {
    setPending(true); onError('')
    try { await command('/reconstruction/' + name, name === 'start' ? options : undefined); setConfirmReset(false) }
    catch (e) { onError((e as Error).message) }
    finally { setPending(false) }
  }
  function number(key: keyof PipelineOptions, label: string, min: number, max: number, step: number) {
    return <label className="field" key={key}>{label}<input type="number" value={options[key] as number} min={min} max={max} step={step} disabled={busy} onChange={e => setOptions(v => ({ ...v, [key]: Number(e.target.value) }))} /></label>
  }
  return <section className="panel pipeline-panel">
    <div className="panel-heading"><div><span className="eyebrow">04 / PROCESSING</span><h2>Detection & reconstruction</h2></div><span className="subtle-badge">{selected} SELECTED</span></div>
    <div className="parameter-grid">{number('threshold', 'DETR confidence', .05, .99, .05)}{number('voxel_size', 'Voxel size (m)', .0005, .1, .001)}{number('depth_min', 'Minimum depth (m)', .001, 100, .05)}{number('depth_max', 'Maximum depth (m)', .01, 1000, .5)}</div>
    <div className="pipeline-options"><label><input type="checkbox" checked={options.create_mesh} disabled={busy} onChange={e => setOptions(v => ({ ...v, create_mesh: e.target.checked }))} />Generate mesh</label><label title="When unchecked, an object outside DETR's known classes may use the explicit user ROI for GrabCut."><input type="checkbox" checked={options.require_detection} disabled={busy} onChange={e => setOptions(v => ({ ...v, require_detection: e.target.checked }))} />Require DETR match</label></div>
    <label className="checkbox-field"><input type="checkbox" checked={options.depth_cleanup} disabled={busy} onChange={e => setOptions(v => ({ ...v, depth_cleanup: e.target.checked }))} />Remove extreme foreground depth outliers</label>
    <details className="settings-details"><summary>Registration & surface parameters</summary><div className="parameter-grid">
      {number('depth_outlier_strength', 'Depth outlier tolerance', 2, 20, 1)}{number('stride', 'Pixel stride', 1, 16, 1)}{number('icp_distance', 'ICP distance (m)', .001, .5, .005)}{number('icp_iterations', 'ICP iterations', 10, 200, 10)}{number('min_fitness', 'Minimum overlap', .1, .95, .05)}{number('normal_radius', 'Normal radius (m)', .002, .5, .005)}{number('outlier_neighbors', 'Outlier neighbors', 5, 100, 5)}{number('outlier_std', 'Outlier deviation', .5, 5, .5)}{number('mesh_radius_factor', 'Mesh radius factor', 1, 10, .5)}
    </div></details>
    <div className="pipeline-actions"><button className="button primary" disabled={!online || busy || !selected} onClick={() => void action('start')}><Scan size={16} />DETECT & reconstruct</button>{state?.busy && <button className="button stop" disabled={pending} onClick={() => void action('cancel')}><Square size={13} />Cancel</button>}</div>
    <div className={'pipeline-status ' + (state?.state === 'failed' ? 'failed' : '')} aria-live="polite"><div><b>{state?.state.replaceAll('_', ' ').toUpperCase() || 'IDLE'}</b><span>{state?.elapsed_seconds.toFixed(1) || '0.0'} s</span></div><progress max={100} value={state?.progress || 0} /><p>{state?.message || 'Capture and select an object viewpoint to begin.'}</p></div>
    {state?.result && <><div className="export-actions"><a className="button secondary" href="/api/reconstruction/export/ply"><Download size={13} />Export PLY</a>{state.result.triangles > 0 && <a className="button secondary" href="/api/reconstruction/export/obj"><Download size={13} />Export OBJ</a>}<a className="download-link" href="/api/integration/isaac" target="_blank" rel="noreferrer">Isaac transfer metadata</a></div><p className="small muted">{state.result.accepted.length} accepted · {state.result.rejected.length} rejected · {state.result.points.toLocaleString()} fused points</p>{state.result.mesh_error && <p className="warning-text small">Mesh unavailable: {state.result.mesh_error}</p>}</>}
    <div className="reset-row">{confirmReset ? <><span>Delete all captured views and reconstruction?</span><button className="text-button danger" disabled={busy} onClick={() => void action('reset')}>Yes, reset session</button><button className="text-button" onClick={() => setConfirmReset(false)}>Keep session</button></> : <button className="text-button" disabled={!online || busy || !state?.views.length} onClick={() => setConfirmReset(true)}><Trash2 size={12} />Reset reconstruction</button>}</div>
    <p className="small muted">Monocular depth is an estimate. Use textured, well-lit objects and overlapping views. Failed registration is excluded from fusion.</p>
  </section>
}

