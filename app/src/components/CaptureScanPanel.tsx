import { useState } from 'react'

import { Camera, Pause, Play, Trash2, Hammer } from 'lucide-react'

import { command } from '../services/api'

import type { ReconstructionState } from '../types'



export function CaptureScanPanel({ state, running, online, onError }: { state: ReconstructionState | null; running: boolean; online: boolean; onError: (s: string) => void }) {

  const [pending, setPending] = useState(false)

  const [confirmDiscard, setConfirmDiscard] = useState(false)

  const [review, setReview] = useState(false)

  const [sharpness, setSharpness] = useState(25)

  const [motion, setMotion] = useState(80)

  const capture = state?.capture

  const geometry = state?.result?.quality

  const included = state?.views.filter(view => view.selected) || []

  const phase = capture?.phase || 'empty'

  const exists = phase !== 'empty'

  const busy = !!state?.busy

  const collecting = phase === 'collecting' || phase === 'preparing'

  const count = capture?.collected || 0

  const target = capture?.target || 1000

  async function act(action: string) {

    setPending(true); onError('')

    try {

      await command('/reconstruction/capture/' + action, action === 'start' ? {

        reference_ids: included.map(view => view.id), minimum_sharpness: sharpness, maximum_motion: motion,

      } : undefined)

      setConfirmDiscard(false)

    } catch (e) { onError((e as Error).message) }

    finally { setPending(false) }

  }

  return <section className="panel capture-workflow" id="capture-set">

    <div className="workflow-title"><div><span className="eyebrow">DETECT → COLLECT → RECONSTRUCT</span><h2>Object capture set</h2></div><span className="subtle-badge">{phase.replaceAll('_', ' ')}</span></div>

    <div className="workflow-layout">

      <div><div className="capture-count"><strong>{count.toLocaleString()}</strong><span>/ {target.toLocaleString()} images</span></div><progress aria-label="Collected images" value={count} max={target} /><p className="small muted">One accepted image every 1.25 s · 20m 50s minimum for 1,000</p></div>

      <div className="workflow-guidance"><p>{capture?.message || 'Add crops of the same object from several angles, then start collecting.'}</p><div className="capture-badges"><span>{exists ? capture?.reference_count ?? included.length : included.length} reference crops</span><span>{((capture?.disk_bytes || 0) / 1024 ** 2).toFixed(0)} MB saved</span>{collecting && capture?.quality && <span className={capture.quality.stable ? 'stable-badge' : 'hold-badge'}>{capture.quality.reason} · {capture.quality.sharpness.toFixed(0)} sharpness</span>}{geometry && <span>{(geometry.median_silhouette_iou * 100).toFixed(0)}% outline agreement · {geometry.camera_span_deg.toFixed(0)}° camera span</span>}{capture?.registered !== undefined && !collecting && <span>{capture.registered} aligned · {capture.rejected || 0} rejected</span>}</div>{capture?.active && !collecting && !!capture.stage_total && <progress aria-label="Reconstruction stage progress" value={capture.processed || 0} max={capture.stage_total} />}</div>

      <div className="capture-buttons">

        {!exists && <button className="button primary" disabled={!online || !running || !included.length || busy || pending} onClick={() => void act('start')}><Camera size={16} />Collect 1,000 images</button>}

        {exists && !capture?.sealed && !capture?.active && <button className="button primary" disabled={!online || !running || busy || pending} onClick={() => void act('resume')}><Play size={15} />Resume collection</button>}

        {capture?.active && <button className="button secondary" disabled={!online || pending} onClick={() => void act('pause')}><Pause size={15} />{collecting ? 'Pause collection' : 'Stop reconstruction'}</button>}

        {exists && !capture?.active && <button className="button primary" disabled={!online || busy || pending || count < 3} onClick={() => void act('build')}><Hammer size={15} />{capture?.sealed ? 'Rebuild saved set' : 'Build saved images now'}</button>}

        {state?.result && <a className="button secondary" href="/api/reconstruction/export/obj">Export mesh OBJ</a>}

      </div>

    </div>

    {!running && !busy && <p className="small muted">Open Camera setup to start the camera. Reconstruction of saved images works with the camera off.</p>}

    {capture?.sealed && <p className="small muted">Capture set sealed. New camera frames cannot modify this reconstruction.</p>}

    <details className="capture-options"><summary>Capture quality, saved images & storage</summary>

      <p className="small muted">Keep the object and background stationary. Move the camera slowly through overlapping views, including high and low angles. Stability selects sharp originals without warping them. Reference crops guide detection; they are never reconstruction images.</p>

      <div className="quality-fields"><label className="field">Minimum sharpness<input type="number" min={5} max={200} value={sharpness} disabled={exists || busy} onChange={e => setSharpness(Number(e.target.value))} /></label><label className="field">Maximum motion (px/s)<input type="number" min={10} max={300} value={motion} disabled={exists || busy} onChange={e => setMotion(Number(e.target.value))} /></label></div>

      <p className="small muted">Temporary files stay in backend/.cache/capture-session (16 GiB limit). Pause and backend restart retain this set. Reconstruct from the fixed set after 1,000 captures, or pause and build a smaller set. Image count alone does not establish full coverage. Camera positions are solved together. Object silhouettes constrain one shared outer surface; depth estimates supply one common approximate scale. Hidden concavities are not measured. The validated mesh is restored after backend restart.</p>

      {capture?.sealed && (phase === 'completed' || phase === 'failed') && <a className="button secondary" href="/api/reconstruction/capture/report" target="_blank" rel="noreferrer">Inspect alignment report</a>}

      {count > 0 && <button className="button secondary" onClick={() => setReview(!review)}>{review ? 'Hide saved images' : 'Review latest saved images'}</button>}

      {review && count > 0 && <div className="saved-strip">{Array.from({ length: Math.min(6, count) }, (_, i) => count - Math.min(6, count) + i).map(index => <a key={index} href={'/api/reconstruction/capture/image/' + index} target="_blank" rel="noreferrer"><img src={'/api/reconstruction/capture/image/' + index} alt={'Saved image ' + (index + 1)} /><span>Image {index + 1}</span></a>)}</div>}

      {exists && !busy && <div className="discard-controls">{confirmDiscard ? <><span>Delete this temporary capture set and its reconstruction?</span><button className="button stop" disabled={pending} onClick={() => void act('discard')}>Delete saved set</button><button className="button secondary" onClick={() => setConfirmDiscard(false)}>Keep it</button></> : <button className="button secondary" disabled={!online || pending} onClick={() => setConfirmDiscard(true)}><Trash2 size={14} />Discard capture set</button>}</div>}

    </details>

  </section>

}

