import { useRef } from 'react'
import { Camera, Maximize2, Radio, VideoOff } from 'lucide-react'
import type { CameraStatus } from '../types'

interface Props { camera: CameraStatus | null; image: string | null; online: boolean; onLoad: () => void }

export function LivePreview({ camera, image, online, onLoad }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const running = online && camera?.state === 'running'
  async function fullscreen() {
    if (document.fullscreenElement) await document.exitFullscreen()
    else await container.current?.requestFullscreen()
  }
  return <section className="preview-panel panel">
    <div className="preview-heading">
      <div><span className="eyebrow">02 / ACQUISITION</span><h2>Live viewport</h2></div>
      <span className={'pill ' + (running ? 'live-pill' : '')}><span className={'status-dot ' + (running ? 'green' : '')} />
        {running ? 'LIVE' : camera?.state === 'starting' ? 'CONNECTING' : 'OFFLINE'}
      </span>
    </div>
    <div ref={container} className="viewport">
      <div className="viewport-label"><Radio size={13} /> {camera?.preview_options?.mode === 'rgb' || !camera?.preview_options ? 'ORIGINAL RGB' : camera.preview_options.mode.replaceAll('_', ' ').toUpperCase()}</div>
      {running && image
        ? <img src={image} alt="Live camera feed" className="camera-image" onLoad={onLoad} onError={onLoad} />
        : <div className="empty-preview">
          <div className="empty-icon">{camera?.state === 'error' ? <VideoOff size={32} strokeWidth={1.3} /> : <Camera size={32} strokeWidth={1.3} />}</div>
          <h3>{!online ? 'Waiting for backend' : camera?.state === 'starting' ? 'Connecting to your camera' : camera?.state === 'error' ? 'Camera unavailable' : 'Your next perspective starts here'}</h3>
          <p>{!online ? 'Start the DepthCloud backend to connect this workspace.' : camera?.state === 'error' ? 'Check the camera connection and source settings, then retry.' : 'Choose a source and start the camera to see a continuous RGB stream.'}</p>
        </div>}
      <div className="frame-corner top-left" /><div className="frame-corner top-right" />
      <div className="frame-corner bottom-left" /><div className="frame-corner bottom-right" />
      <div className="viewport-bottom">
        <span>{running ? '#' + String(camera.latest_sequence ?? 0).padStart(6, '0') : 'NO FRAME'}</span>
        <button className="icon-button" onClick={() => { void fullscreen().catch(() => {}) }} aria-label="Toggle fullscreen preview"><Maximize2 size={17} /></button>
      </div>
    </div>
    <div className="preview-footer">
      <span><span className={'status-dot ' + (running ? 'green' : '')} />{running ? 'Receiving live frames' : 'No active stream'}</span>
      <span className="mono">{camera?.actual.width ? camera.actual.width + ' × ' + camera.actual.height : '— × —'} <span className="separator">/</span> {running ? camera.capture_fps.toFixed(1) : '—'} fps</span>
    </div>
  </section>
}

