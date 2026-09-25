import { useRef, useState } from 'react'
import type { PointerEvent } from 'react'
import { Crop, X } from 'lucide-react'
import type { FrozenFrame, ROI } from '../types'

interface Props { frame: FrozenFrame; initialLabel: string; onClose: () => void; onCapture: (roi: ROI, label: string) => Promise<void> }
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

export function RoiEditor({ frame, initialLabel, onClose, onCapture }: Props) {
  const [roi, setRoi] = useState<ROI>({ x: .2, y: .15, width: .6, height: .7 })
  const [label, setLabel] = useState(initialLabel)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const drag = useRef<{ kind: string; start: [number, number]; roi: ROI } | null>(null)
  function position(event: PointerEvent<SVGSVGElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect()
    return [clamp((event.clientX - rect.left) / rect.width, 0, 1), clamp((event.clientY - rect.top) / rect.height, 0, 1)]
  }
  function down(event: PointerEvent<SVGSVGElement>) {
    if (busy) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const kind = (event.target as SVGElement).dataset.drag || 'draw'
    drag.current = { kind, start: position(event), roi: { ...roi } }
  }
  function move(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current
    if (!active) return
    const [x, y] = position(event)
    const [sx, sy] = active.start
    const previous = active.roi
    if (active.kind === 'move') {
      setRoi({ ...previous, x: clamp(previous.x + x - sx, 0, 1 - previous.width), y: clamp(previous.y + y - sy, 0, 1 - previous.height) })
      return
    }
    let x0 = previous.x, y0 = previous.y, x1 = x0 + previous.width, y1 = y0 + previous.height
    if (active.kind === 'draw') { x0 = Math.min(x, sx); y0 = Math.min(y, sy); x1 = Math.max(x, sx); y1 = Math.max(y, sy) }
    else {
      if (active.kind.includes('w')) x0 = Math.min(x, x1 - .01)
      if (active.kind.includes('e')) x1 = Math.max(x, x0 + .01)
      if (active.kind.includes('n')) y0 = Math.min(y, y1 - .01)
      if (active.kind.includes('s')) y1 = Math.max(y, y0 + .01)
    }
    x0 = clamp(x0, 0, .99); y0 = clamp(y0, 0, .99)
    setRoi({ x: x0, y: y0, width: clamp(x1 - x0, .01, 1 - x0), height: clamp(y1 - y0, .01, 1 - y0) })
  }
  async function capture() {
    setBusy(true); setError('')
    try { await onCapture(roi, label) } catch (e) { setError((e as Error).message); setBusy(false) }
  }
  const handles: [string, number, number][] = [['nw', roi.x, roi.y], ['ne', roi.x + roi.width, roi.y], ['sw', roi.x, roi.y + roi.height], ['se', roi.x + roi.width, roi.y + roi.height]]
  return <div className="modal-backdrop"><section className="roi-dialog" role="dialog" aria-modal="true" aria-label="Select object region">
    <header><div><span className="eyebrow">FROZEN ORIGINAL / FRAME {frame.sequence_id}</span><h2><Crop size={18} /> Select your object</h2></div><button className="icon-button" onClick={onClose} disabled={busy} aria-label="Close crop editor"><X size={20} /></button></header>
    <p className="small muted">Drag to draw. Move the box or drag a corner to resize. Live capture continues independently.</p>
    <div className="roi-stage" style={{ aspectRatio: frame.width + '/' + frame.height }}>
      <img src={'/api/reconstruction/snapshot/' + frame.id} alt="Frozen original for cropping" draggable={false} />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Interactive object crop" onPointerDown={down} onPointerMove={move} onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }}>
        <defs><mask id="roi-shade"><rect width="1" height="1" fill="white" /><rect x={roi.x} y={roi.y} width={roi.width} height={roi.height} fill="black" /></mask></defs>
        <rect width="1" height="1" fill="#030a05" opacity=".55" mask="url(#roi-shade)" />
        <rect data-drag="move" x={roi.x} y={roi.y} width={roi.width} height={roi.height} fill="transparent" stroke="#cefa8f" strokeWidth=".003" style={{ cursor: 'move' }} />
        {handles.map(([kind, x, y]) => <rect data-drag={kind} key={kind} x={x - .008} y={y - .01} width=".016" height=".02" fill="#cefa8f" stroke="#23351c" strokeWidth=".002" style={{ cursor: kind === 'nw' || kind === 'se' ? 'nwse-resize' : 'nesw-resize' }} />)}
      </svg>
    </div>
    <div className="roi-fields">
      <label className="field">Reference label (optional angle name)<input value={label} maxLength={64} onChange={e => setLabel(e.target.value)} /></label>
      {(['x', 'y', 'width', 'height'] as const).map(key => <label className="field" key={key}>ROI {key} %<input type="number" min={0} max={100} step={.1} value={Number((roi[key] * 100).toFixed(1))} onChange={e => setRoi(v => ({ ...v, [key]: Number(e.target.value) / 100 }))} /></label>)}
    </div>
    <div className="roi-footer"><span className="mono">{Math.round(roi.width * frame.width)} × {Math.round(roi.height * frame.height)} native pixels · original RGB retained</span><button className="button primary" disabled={busy || roi.width <= 0 || roi.height <= 0 || roi.x + roi.width > 1.000001 || roi.y + roi.height > 1.000001} onClick={() => void capture()}>{busy ? 'Saving…' : 'Confirm & capture view'}</button></div>
    {error && <p role="alert" className="inline-error">{error}</p>}
  </section></div>
}

