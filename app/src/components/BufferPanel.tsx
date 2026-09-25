import { useState } from 'react'
import { ArrowRight, Layers3 } from 'lucide-react'
import { command } from '../services/api'
import type { CameraStatus } from '../types'

export function BufferPanel({ camera, online, onError }: { camera: CameraStatus | null; online: boolean; onError: (value: string) => void }) {
  const [draft, setDraft] = useState<{ capacity?: number; automatic?: boolean }>({})
  const capacity = draft.capacity ?? camera?.current_buffer_size ?? 10
  const automatic = draft.automatic ?? camera?.automatic_buffer ?? false
  const [busy, setBusy] = useState(false)
  async function apply() {
    setBusy(true)
    onError('')
    try { await command('/camera/buffer', { capacity, automatic }, 'PATCH'); setDraft({}) }
    catch (error) { onError((error as Error).message) }
    finally { setBusy(false) }
  }
  const slots = camera?.buffer_sequence_ids || []
  return <section className="panel buffer-panel">
    <div className="buffer-header"><div><span className="eyebrow">03 / FRAME MEMORY</span><h2>Rolling buffer <span className="subtle-badge">FIFO</span></h2></div><Layers3 size={18} /></div>
    <div className="buffer-description"><span>Oldest out. Latest retained.</span><span className="mono">{camera?.buffer_fill ?? 0} / {camera?.current_buffer_size ?? capacity} frames</span></div>
    <div className="buffer-track" aria-label="Recent buffered frame sequence IDs">
      {slots.length ? slots.map((id, i) => <div key={id} className={'frame-slot filled ' + (i === slots.length - 1 ? 'newest' : '')}><span>FRAME</span><b>{String(id).padStart(4, '0')}</b></div>)
        : Array.from({ length: 8 }, (_, i) => <div key={i} className="frame-slot"><span>EMPTY</span><b>—</b></div>)}
    </div>
    <div className="buffer-direction"><span>OLDEST</span><div /><ArrowRight size={12} /><span>NEWEST</span></div>
    <div className="buffer-controls">
      <label className="field compact">Capacity<input type="number" min="1" max={camera?.maximum_buffer_size || 128} value={capacity} disabled={automatic} onChange={e => setDraft(previous => ({ ...previous, capacity: Number(e.target.value) }))} /></label>
      <label className="checkbox-field"><input type="checkbox" checked={automatic} onChange={e => setDraft(previous => ({ ...previous, automatic: e.target.checked }))} />Automatic sizing</label>
      <button className="button secondary" disabled={!online || busy || capacity < 1 || capacity > 128} onClick={apply}>{busy ? 'Applying…' : 'Apply'}</button>
    </div>
    <p className="small muted buffer-note">{camera?.automatic_buffer ? 'Automatic: capacity follows measured capture rate, preview latency, and available RAM.' : 'Manual capacity, with a hard memory limit. Stopping or disconnecting flushes all transient frames.'}</p>
  </section>
}


