import { useState } from 'react'
import { command } from '../services/api'
import type { ReconstructionState } from '../types'

export function CalibrationPanel({ state, online, onError }: { state: ReconstructionState | null; online: boolean; onError: (s: string) => void }) {
  const [board, setBoard] = useState({ columns: 9, rows: 6, square_size: .025 })
  const [json, setJson] = useState('')
  const [pending, setPending] = useState(false)
  const calibration = state?.calibration
  const disabled = !online || state?.busy || pending
  async function act(path: string, body?: unknown, method = 'POST') {
    setPending(true)
    try { await command('/calibration' + path, body, method) }
    catch (e) { onError((e as Error).message) }
    finally { setPending(false) }
  }
  function importCalibration() {
    try { const payload: unknown = JSON.parse(json); void act('', payload, 'PUT') }
    catch { onError('Enter valid calibration JSON with fx, fy, cx, cy, width, height and optional distortion coefficients.') }
  }
  function download() {
    if (!calibration?.intrinsics) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(calibration.intrinsics, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'depthcloud-calibration.json'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <details className="panel calibration-panel"><summary><span>Camera calibration</span><span className="subtle-badge">{calibration?.intrinsics ? calibration.intrinsics.source.toUpperCase() : 'ESTIMATED INTRINSICS'}</span></summary>
    <div className="calibration-body"><p className="small muted">Capture at least 8 sharp checkerboard views at different positions and tilts. Counts below refer to inner corners; square size is measured in meters.</p>
      <div className="parameter-grid">{([['columns', 'Inner columns', 3, 20, 1], ['rows', 'Inner rows', 3, 20, 1], ['square_size', 'Square size (m)', .001, 1, .001]] as const).map(([key, label, min, max, step]) => <label className="field" key={key}>{label}<input type="number" value={board[key]} min={min} max={max} step={step} disabled={disabled} onChange={e => setBoard(v => ({ ...v, [key]: Number(e.target.value) }))} /></label>)}</div>
      <div className="export-actions"><button className="button secondary" disabled={disabled} onClick={() => void act('/capture', board)}>Capture checkerboard ({calibration?.samples || 0})</button><button className="button secondary" disabled={disabled || (calibration?.samples || 0) < 8} onClick={() => void act('/solve')}>Solve calibration</button><button className="text-button" disabled={disabled} onClick={() => void act('', undefined, 'DELETE')}>Clear calibration</button></div>
      {calibration?.intrinsics && <><p className="small mono">fx {calibration.intrinsics.fx.toFixed(2)} · fy {calibration.intrinsics.fy.toFixed(2)} · {calibration.intrinsics.width} × {calibration.intrinsics.height}{calibration.intrinsics.rms !== null ? ' · RMS ' + calibration.intrinsics.rms.toFixed(3) + ' px' : ''}</p><button className="text-button" onClick={download}>Download calibration JSON</button></>}
      <label className="field calibration-json">Import measured intrinsics (JSON)<textarea value={json} onChange={e => setJson(e.target.value)} rows={3} placeholder="Paste calibration JSON containing fx, fy, cx, cy, width, height and distortion." /></label><button className="button secondary" disabled={disabled || !json.trim()} onClick={importCalibration}>Import intrinsics</button>
      <p className="small muted">Calibration is kept for this backend session; download JSON to retain it. Without calibration, Depth Pro focal length estimates are explicitly labeled.</p>
    </div>
  </details>
}

