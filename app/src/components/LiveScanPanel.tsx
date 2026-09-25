import { useState } from 'react'
import { Scan, Square } from 'lucide-react'
import { command } from '../services/api'
import type { ReconstructionState } from '../types'

export function LiveScanPanel({ state, running, online, onError }: { state: ReconstructionState | null; running: boolean; online: boolean; onError: (s: string) => void }) {

  const [scanMode, setScanMode] = useState<'moving_camera' | 'rotating_object'>('moving_camera')
  const [pending, setPending] = useState(false)
  const [colorFilter, setColorFilter] = useState(true)
  const [geometryTracking, setGeometryTracking] = useState(false)
  const [resolution, setResolution] = useState(192)
  const views = state?.views || []
  const included = views.filter(view => view.selected)
  const live = state?.live
  const scanning = !!state?.busy && !!live
  async function start() {
    setPending(true); onError('')
    try { await command('/reconstruction/live/start', { reference_ids: included.map(view => view.id), scan_mode: scanMode, color_filter: colorFilter, allow_geometry_tracking: scanMode === 'rotating_object' && geometryTracking, volume_resolution: resolution, interval_seconds: .5, pipeline: { voxel_size: .002, normal_radius: .015, icp_distance: .015, min_fitness: .65, depth_outlier_strength: 5 } }) }
    catch (e) { onError((e as Error).message) }
    finally { setPending(false) }
  }
  async function stop() {
    setPending(true)
    try { await command('/reconstruction/live/stop') } catch (e) { onError((e as Error).message) } finally { setPending(false) }
  }
  return <section className="panel live-scan-panel">
    <div className="panel-heading"><div><span className="eyebrow">04 / CONTINUOUS GENERATION</span><h2>Continuous object scan</h2></div><span className="subtle-badge">{scanning ? 'LIVE' : 'READY'}</span></div>
    <div className="live-scan-content">
      <p className="small muted">Include crops of the same object from different angles in the gallery. All checked crops guide identification across the full camera image. Only new live frames build the surface. Choose which part of the setup will move before starting.</p>
      <label className="field">Scanning motion<select value={scanMode} disabled={!!state?.busy || pending} onChange={e => setScanMode(e.target.value as 'moving_camera' | 'rotating_object')}><option value="moving_camera">Move camera · stationary object and background</option><option value="rotating_object">Rotate object · fixed camera</option></select></label>
      <p className="small muted">{scanMode === 'moving_camera' ? 'Keep the object and surrounding scene still. Move the camera slowly around it. Background features estimate camera motion; only the matched object is fused. Keep textured surroundings visible.' : 'Keep the camera fixed and rotate the object slowly. Tracking needs distinctive features on the object; scene motion cannot substitute for object pose.'}</p>
      <div className="field" aria-label="Object reference set"><b>Object reference set · {included.length} included</b><span className="small muted">{included.length ? included.map(view => view.label).join(' · ') : 'Capture object crops, then check the angles to include in the gallery.'}</span></div>
      <div className="pipeline-actions"><button className="button primary" disabled={!online || !running || !included.length || !!state?.busy || pending} onClick={() => void start()}><Scan size={15} />Start generation</button>{scanning && <button className="button stop" disabled={pending || live?.stopping} onClick={() => void stop()}><Square size={13} />{live?.stopping ? 'Stopping…' : 'Stop generation'}</button>}</div>
      {!running && <p className="small muted">Start the camera before generation.</p>}
      {live && <><div className="pipeline-status" aria-live="polite"><div><b>{live.tracking.replaceAll('_', ' ').toUpperCase()}</b><span>{live.processing_fps.toFixed(1)} processed fps</span></div><p>{state?.message}</p></div><div className="live-metrics"><span>{live.integrated} integrated</span><span>{live.rejected} rejected</span><span>{live.skipped} unchanged</span><span>{live.surface_updates} mesh updates</span></div><p className="small muted">Observed directions: {live.directions.join(', ') || 'waiting for first live frame'}. {live.voxel_size ? `Surface spacing ${(live.voxel_size * 1000).toFixed(1)} mm (estimated).` : ''}</p>
      {live.scan_mode === 'moving_camera' && live.registration && <p className="small muted">Camera {live.camera_tracking}. Scene support: {live.registration.scene_inliers ?? 0} inliers · reprojection {live.registration.reprojection_px?.toFixed(2) ?? '—'} px · depth correction {live.registration.depth_scale?.toFixed(3) ?? '—'}×{live.registration.relocalized ? ' · recovered from an earlier keyframe' : ''}</p>}
      <p className="small muted">{live.prepared_references} / {live.reference_count} references prepared. {live.matched_reference ? `Best appearance match: ${live.matched_reference.label} · similarity ${live.matched_reference.score.toFixed(2)}. All included references remain active.` : 'No current appearance match.'}</p>
      {live.preview_sequence !== null && <><img className="live-mask-preview" src={'/api/reconstruction/live/image?frame=' + live.preview_sequence} alt={'Processed live frame ' + live.preview_sequence + '; unmatched background is darkened'} /><p className="small muted">Live frame #{live.preview_sequence}. {live.tracking === 'uncertain' ? 'Rejected; surface unchanged.' : 'Highlighted pixels show the matched foreground; fusion also requires a reliable pose.'}</p></>}
      <p className="small muted">Reference names identify your saved examples, not measured object angles. Appearance matching uses local model features, color and silhouette; similar objects and occlusion can remain ambiguous. Hidden surfaces require overlapping live observations.</p></>}
      <details className="settings-details"><summary>Tracking & surface settings</summary><label className="checkbox-field"><input type="checkbox" checked={colorFilter} disabled={!!state?.busy} onChange={e => setColorFilter(e.target.checked)} />Use matched reference colors for the foreground mask</label><label className="checkbox-field"><input type="checkbox" checked={geometryTracking} disabled={!!state?.busy || scanMode === 'moving_camera'} onChange={e => setGeometryTracking(e.target.checked)} />Allow geometry-only tracking (ambiguous on symmetric objects)</label><p className="warning-text small">Rotating-object mode needs features on the object; smooth or symmetric objects need nonrepeating removable markers. Moving-camera mode uses the stationary scene instead. Geometry-only tracking can confuse rotations and cannot establish reliable side coverage.</p><label className="field">Surface volume resolution<select value={resolution} disabled={!!state?.busy} onChange={e => setResolution(Number(e.target.value))}><option value={128}>128 · lower memory</option><option value={192}>192 · balanced</option><option value={256}>256 · finer / more memory</option></select></label></details>
      {state?.result && <div className="export-actions"><a className="button secondary" href="/api/reconstruction/export/ply">Export PLY</a>{state.result.triangles > 0 && <a className="button secondary" href="/api/reconstruction/export/obj">Export mesh OBJ</a>}</div>}
      <p className="small muted">Start generation replaces the previous surface. Stop retains the current surface and keeps the camera running. Live updates run at measured processing speed, not the camera's requested FPS.</p>
    </div>
  </section>
}
