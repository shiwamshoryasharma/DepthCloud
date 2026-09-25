import { useState } from 'react'
import { command } from '../services/api'
import type { PreviewOptions } from '../types'

const defaults: PreviewOptions = { mode: 'rgb', saturation: 1.5, contrast: 1.1, brightness: 0, block_size: 12, mono_threshold: 127, ir_palette: 'inferno' }
const modes: [PreviewOptions['mode'], string][] = [['rgb', 'RGB'], ['vivid', 'Vivid'], ['mono', 'Mono'], ['chrome', 'Chrome'], ['pixelated', 'Pixelated'], ['pseudo_ir', 'Pseudo IR'], ['grayscale', 'Gray']]

export function PreviewControls({ current, online, onError }: { current?: PreviewOptions; online: boolean; onError: (s: string) => void }) {
  const [draft, setDraft] = useState<Partial<PreviewOptions>>({})
  const [busy, setBusy] = useState(false)
  const options = { ...defaults, ...current, ...draft }
  async function apply(mode = options.mode) {
    setBusy(true)
    try { await command('/camera/preview', { ...options, mode }, 'PATCH'); setDraft({}) }
    catch (e) { onError((e as Error).message) }
    finally { setBusy(false) }
  }
  return <div className="representation-controls">
    <div className="mode-tabs" aria-label="Image representations">{modes.map(([mode, label]) => <button key={mode} aria-pressed={current?.mode === mode || (!current && mode === 'rgb')} disabled={!online || busy} onClick={() => void apply(mode)}>{label}</button>)}</div>
    {current?.mode === 'pseudo_ir' && <p className="small muted">Pseudo IR is a color visualization of RGB brightness, not thermal or infrared sensor data.</p>}
    <details className="settings-details"><summary>Image transform settings</summary><div className="parameter-grid">
      {([['saturation', 'Saturation', 0, 3, .1], ['contrast', 'Contrast', .2, 3, .1], ['brightness', 'Brightness', -100, 100, 1], ['block_size', 'Pixel block size', 2, 80, 1], ['mono_threshold', 'Mono threshold', 0, 255, 1]] as const).map(([key, label, min, max, step]) => <label className="field" key={key}>{label}<input type="number" value={options[key]} min={min} max={max} step={step} onChange={e => setDraft(v => ({ ...v, [key]: Number(e.target.value) }))} /></label>)}
      <label className="field">Pseudo-IR palette<select value={options.ir_palette} onChange={e => setDraft(v => ({ ...v, ir_palette: e.target.value as PreviewOptions['ir_palette'] }))}><option value="inferno">Inferno</option><option value="turbo">Turbo</option></select></label>
    </div><button className="button secondary" disabled={!online || busy} onClick={() => void apply()}>Apply image settings</button></details>
  </div>
}

