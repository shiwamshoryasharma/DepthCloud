import { useState } from 'react'
import { Camera, ChevronDown, LoaderCircle, Play, ScanLine, Square } from 'lucide-react'
import { command } from '../services/api'
import type { CameraDevice, CameraStatus } from '../types'

interface Props { camera: CameraStatus | null; online: boolean; onError: (message: string) => void; onRecord: (message: string) => void }

export function CameraControls({ camera, online, onError, onRecord }: Props) {
  const [devices, setDevices] = useState<CameraDevice[]>([])
  const [index, setIndex] = useState(0)
  const [resolution, setResolution] = useState('1280x720')
  const [fps, setFps] = useState(30)
  const [backend, setBackend] = useState('auto')
  const [busy, setBusy] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [scanned, setScanned] = useState(false)
  const active = ['starting', 'running', 'stopping'].includes(camera?.state || '')
  const locked = busy || scanning || active || !online

  async function discover() {
    setScanning(true)
    onError('')
    try {
      const data = await command<{ devices: CameraDevice[]; timed_out_indices: number[]; probe_errors: { index: number; exit_code: number }[] }>('/camera/discover?backend=' + encodeURIComponent(backend))
      setDevices(data.devices)
      setScanned(true)
      if (data.devices.length) setIndex(data.devices[0].index)
      onRecord('Discovery complete · ' + data.devices.length + ' camera(s) found')
      if (data.probe_errors.length) onError('Camera discovery encountered a driver/process error. Check the backend log; manual selection is still available.')
      if (data.timed_out_indices.length) onError('Some camera probes timed out. You can still enter an index manually or try another backend.')
    } catch (error) { onError((error as Error).message) }
    finally { setScanning(false) }
  }

  async function toggle() {
    setBusy(true)
    onError('')
    try {
      if (active) await command('/camera/stop')
      else {
        const [width, height] = resolution.split('x').map(Number)
        await command('/camera/start', { index, width, height, fps, backend })
      }
    } catch (error) { onError((error as Error).message) }
    finally { setBusy(false) }
  }

  return <section className="panel source-panel">
    <div className="section-title"><span className="eyebrow">01 / SOURCE</span><Camera size={16} /></div>
    <h2>Camera input</h2>
    <p className="muted small">Connect a USB camera to begin acquisition.</p>
    <button className="button secondary scan-button" onClick={discover} disabled={locked}>
      {scanning ? <LoaderCircle size={15} className="spin" /> : <ScanLine size={15} />}
      {scanning ? 'Discovering cameras…' : 'Discover cameras'}
    </button>
    {devices.length > 0 && <label className="field">Detected device
      <select value={index} onChange={e => setIndex(Number(e.target.value))} disabled={locked}>
        {devices.map(device => <option key={device.index} value={device.index}>{device.label} · {device.backend}</option>)}
      </select>
    </label>}
    {scanned && devices.length === 0 && <p className="small warning-text">No camera responded. Check USB access, or try a manual index.</p>}
    <label className="field">Camera index
      <input type="number" min="0" max="32" value={index} disabled={locked} onChange={e => setIndex(Number(e.target.value))} />
    </label>
    <label className="field">Requested resolution
      <select value={resolution} onChange={e => setResolution(e.target.value)} disabled={locked}>
        <option value="640x480">640 × 480 · SD</option>
        <option value="1280x720">1280 × 720 · HD</option>
        <option value="1920x1080">1920 × 1080 · FHD</option>
      </select>
    </label>
    <label className="field">Requested frame rate
      <select value={fps} onChange={e => setFps(Number(e.target.value))} disabled={locked}>
        {[15, 24, 30, 60].map(value => <option key={value} value={value}>{value} fps</option>)}
      </select>
    </label>
    <details className="advanced"><summary>Capture backend <ChevronDown size={14} /></summary>
      <label className="field"><span className="sr-only">Capture backend</span>
        <select value={backend} onChange={e => setBackend(e.target.value)} disabled={locked}>
          <option value="auto">Automatic</option><option value="dshow">DirectShow</option>
          <option value="msmf">Media Foundation</option><option value="any">OpenCV default</option>
        </select>
      </label>
      <p className="small muted">Use another backend if the device cannot open. The camera may negotiate a different resolution or FPS.</p>
    </details>
    <button className={'button ' + (active ? 'stop' : 'primary')} onClick={toggle}
      disabled={!online || busy || scanning || camera?.state === 'stopping'}>
      {busy || camera?.state === 'starting' ? <LoaderCircle size={16} className="spin" /> : active ? <Square size={14} /> : <Play size={16} />}
      {active ? 'Stop camera' : 'Start camera'}
    </button>
    <div className="source-note"><span className={'status-dot ' + (active ? 'green' : '')} />
      {camera?.state === 'running' ? 'Live capture · RAM only' : 'Capture starts only on request'}
    </div>
  </section>
}



