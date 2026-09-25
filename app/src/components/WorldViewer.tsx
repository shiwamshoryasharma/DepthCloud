import { Component, useEffect, useMemo, useRef, useState } from 'react'



import type { ReactNode } from 'react'



import { Canvas, useFrame, useThree } from '@react-three/fiber'



import { Box3, Color, SRGBColorSpace, BufferGeometry, Float32BufferAttribute, InterleavedBuffer, InterleavedBufferAttribute, Uint32BufferAttribute, Vector3, MOUSE, TOUCH } from 'three'



import { OrbitControls } from 'three/addons/controls/OrbitControls.js'



import { Box, Expand, RotateCcw } from 'lucide-react'



import type { ReconstructionResult } from '../types'







interface GeometryData { cloud: Float32Array; vertices: Float32Array | null; triangles: Uint32Array | null }



class ViewerBoundary extends Component<{ children: ReactNode }, { error: string | null }> {



  state = { error: null as string | null }



  static getDerivedStateFromError(error: Error) { return { error: error.message } }



  render() { return this.state.error ? <div className="viewer-failure">3D renderer unavailable: {this.state.error}. PLY/OBJ export remains available.</div> : this.props.children }



}



function geometryFrom(data: Float32Array) {



  const geometry = new BufferGeometry()



  // Camera/export colors are sRGB; Three vertex attributes must be linear.
  const linear = data.slice()
  const color = new Color()
  for (let i = 0; i < linear.length; i += 9) {
    color.setRGB(linear[i + 3], linear[i + 4], linear[i + 5], SRGBColorSpace)
    linear[i + 3] = color.r; linear[i + 4] = color.g; linear[i + 5] = color.b
  }
  const buffer = new InterleavedBuffer(linear, 9)



  geometry.setAttribute('position', new InterleavedBufferAttribute(buffer, 3, 0))



  geometry.setAttribute('color', new InterleavedBufferAttribute(buffer, 3, 3))



  geometry.setAttribute('normal', new InterleavedBufferAttribute(buffer, 3, 6))



  geometry.computeBoundingSphere()



  return geometry



}



function Cloud({ data, pointSize, density }: { data: Float32Array; pointSize: number; density: number }) {



  const geometry = useMemo(() => geometryFrom(data), [data])



  useEffect(() => () => geometry.dispose(), [geometry])



  useEffect(() => {



    const count = data.length / 9



    const step = Math.max(1, Math.round(100 / density))



    geometry.setIndex(step === 1 ? null : new Uint32BufferAttribute(Array.from({ length: Math.ceil(count / step) }, (_, i) => i * step), 1))



  }, [geometry, data, density])



  return <points geometry={geometry}><pointsMaterial size={pointSize} vertexColors sizeAttenuation toneMapped={false} /></points>



}



function Surface({ data, indices, opacity, wireframe }: { data: Float32Array; indices: Uint32Array; opacity: number; wireframe: boolean }) {



  const geometry = useMemo(() => { const value = geometryFrom(data); value.setIndex(new Uint32BufferAttribute(indices, 1)); return value }, [data, indices])



  useEffect(() => () => geometry.dispose(), [geometry])



  return <mesh geometry={geometry}><meshStandardMaterial vertexColors roughness={.85} metalness={0} opacity={opacity} transparent={opacity < 1} depthWrite={opacity >= 1} wireframe={wireframe} side={2} /></mesh>



}



function Normals({ data, length }: { data: Float32Array; length: number }) {



  const geometry = useMemo(() => {



    const positions: number[] = []



    const step = Math.max(1, Math.ceil(data.length / 9 / 800))



    for (let index = 0; index < data.length; index += step * 9) {



      positions.push(data[index], data[index + 1], data[index + 2], data[index] + data[index + 6] * length, data[index + 1] + data[index + 7] * length, data[index + 2] + data[index + 8] * length)



    }



    const value = new BufferGeometry()



    value.setAttribute('position', new Float32BufferAttribute(positions, 3))



    return value



  }, [data, length])



  useEffect(() => () => geometry.dispose(), [geometry])



  return <lineSegments geometry={geometry}><lineBasicMaterial color="#a9df88" /></lineSegments>



}



/* oxlint-disable react/immutability -- R3F camera and OrbitControls are imperative Three.js resources, not React state. */



function Controls({ bounds, fitSignal, navigation, viewDirection, hasGeometry }: { bounds: Box3; fitSignal: number; navigation: string; viewDirection: string; hasGeometry: boolean }) {



  const { camera, gl, invalidate } = useThree()



  const controls = useRef<OrbitControls | null>(null)



  const fitted = useRef(false)



  const lastFit = useRef(-1)



  useEffect(() => {



    // Create inside the effect so StrictMode cleanup cannot leave a disposed memoized instance.



    const value = new OrbitControls(camera, gl.domElement)



    controls.current = value



    value.enableDamping = true



    value.dampingFactor = .09



    value.screenSpacePanning = true



    value.panSpeed = .8



    value.rotateSpeed = .65



    value.zoomSpeed = .8



    value.touches.ONE = TOUCH.ROTATE



    value.touches.TWO = TOUCH.DOLLY_PAN



    const redraw = () => invalidate()



    const preventMenu = (event: Event) => event.preventDefault()



    gl.domElement.addEventListener('contextmenu', preventMenu)



    value.addEventListener('change', redraw)



    invalidate()



    return () => {



      gl.domElement.removeEventListener('contextmenu', preventMenu)



      value.removeEventListener('change', redraw)



      value.dispose()



      controls.current = null



      fitted.current = false



    }



  }, [camera, gl, invalidate])



  useEffect(() => {



    const value = controls.current



    if (!value) return



    value.mouseButtons.LEFT = navigation === 'pan' ? MOUSE.PAN : MOUSE.ROTATE



    value.mouseButtons.MIDDLE = MOUSE.DOLLY



    value.mouseButtons.RIGHT = MOUSE.PAN



    value.touches.ONE = navigation === 'pan' ? TOUCH.PAN : TOUCH.ROTATE



  }, [navigation])



  useFrame(() => { if (controls.current?.update()) invalidate() })



  useEffect(() => {



    const value = controls.current



    if (!value) return



    if (!hasGeometry) fitted.current = false



    // A new mesh revision must not reset the user's pan/zoom/orbit.



    if (fitted.current && lastFit.current === fitSignal) return



    const center = bounds.getCenter(new Vector3())



    const diagonal = Math.max(.15, bounds.getSize(new Vector3()).length())



    const views: Record<string, number[]> = { front: [0, 0, 1.4], back: [0, 0, -1.4], top: [0, 1.4, .001], bottom: [0, -1.4, .001], left: [-1.4, 0, 0], right: [1.4, 0, 0], angle: [.75, .45, 1.15] }



    const direction = views[viewDirection] || views.angle



    value.target.copy(center)



    camera.up.set(0, 1, 0)



    camera.position.copy(center).add(new Vector3(...direction as [number, number, number]).multiplyScalar(diagonal))



    camera.near = Math.max(.0001, diagonal / 1000)



    camera.far = Math.max(100, diagonal * 100)



    camera.updateProjectionMatrix()



    value.minDistance = diagonal / 100



    value.maxDistance = diagonal * 30



    value.update()



    fitted.current = hasGeometry



    lastFit.current = fitSignal



    invalidate()



  }, [bounds, fitSignal, camera, invalidate, viewDirection, hasGeometry])



  return null



}



/* oxlint-enable react/immutability */



export function WorldViewer({ result }: { result: ReconstructionResult | null }) {



  const [loaded, setLoaded] = useState<{ revision: number | undefined; geometry: GeometryData | null; error: string }>({ revision: undefined, geometry: null, error: '' })



  const [mode, setMode] = useState('auto')



  const [navigation, setNavigation] = useState('orbit')



  const [viewDirection, setViewDirection] = useState('angle')



  const [grid, setGrid] = useState(true)



  const [axes, setAxes] = useState(true)



  const [normals, setNormals] = useState(false)



  const [pointSize, setPointSize] = useState(.004)



  const [density, setDensity] = useState(100)



  const [opacity, setOpacity] = useState(1)



  const [wireframe, setWireframe] = useState(false)



  const [fitSignal, setFitSignal] = useState(0)



  const revision = result?.revision



  const geometry = result ? loaded.geometry : null



  const error = loaded.revision === revision ? loaded.error : ''



  const loading = revision !== undefined && loaded.revision !== revision



  const count = result?.points || 0, vertices = result?.vertices || 0, triangles = result?.triangles || 0



  useEffect(() => {



    const abort = new AbortController()



    if (revision === undefined) return () => abort.abort()



    async function fetchBuffer(kind: string) {



      const response = await fetch('/api/reconstruction/geometry/' + kind + '?revision=' + revision, { signal: abort.signal })



      if (!response.ok) throw new Error('Geometry is unavailable or changed. Wait for reconstruction to complete.')



      return response.arrayBuffer()



    }



    async function load() {



      try {



        const [cloudBuffer, meshBuffer] = await Promise.all([fetchBuffer('cloud'), triangles ? fetchBuffer('mesh') : Promise.resolve(null)])



        if (cloudBuffer.byteLength !== count * 9 * 4) throw new Error('Point-cloud payload size mismatch.')



        if (meshBuffer && meshBuffer.byteLength !== vertices * 9 * 4 + triangles * 3 * 4) throw new Error('Mesh payload size mismatch.')



        if (!abort.signal.aborted) setLoaded({ revision, error: '', geometry: { cloud: new Float32Array(cloudBuffer), vertices: meshBuffer ? new Float32Array(meshBuffer, 0, vertices * 9) : null, triangles: meshBuffer ? new Uint32Array(meshBuffer, vertices * 9 * 4, triangles * 3) : null } })



      } catch (e) { if (!abort.signal.aborted) setLoaded({ revision, geometry: null, error: (e as Error).message }) }



    }



    void load()



    return () => abort.abort()



  }, [revision, count, vertices, triangles])



  const xmin = result?.bounds[0][0] ?? -.5, ymin = result?.bounds[0][1] ?? -.5, zmin = result?.bounds[0][2] ?? -.5



  const xmax = result?.bounds[1][0] ?? .5, ymax = result?.bounds[1][1] ?? .5, zmax = result?.bounds[1][2] ?? .5



  const bounds = useMemo(() => new Box3(new Vector3(xmin, -ymax, -zmax), new Vector3(xmax, -ymin, -zmin)), [xmin, ymin, zmin, xmax, ymax, zmax])



  const diagonal = Math.max(.1, bounds.getSize(new Vector3()).length())



  const center = bounds.getCenter(new Vector3())



  const effectiveMode = mode === 'auto' ? (geometry?.vertices ? 'mesh' : 'points') : mode



  const showCloud = effectiveMode !== 'mesh' || !geometry?.vertices



  return <section className="panel world-panel">



    <div className="panel-heading"><div><span className="eyebrow">03 / RECONSTRUCTION WORLD</span><h2>3D viewport</h2></div><span className="pill">{geometry ? 'MEASURED INPUT · ESTIMATED GEOMETRY' : 'READY'}</span></div>



    <div className="world-viewport" aria-label="Interactive 3D reconstruction viewport">



      <ViewerBoundary><Canvas frameloop="demand" dpr={[1, 1.5]} camera={{ position: [1.5, 1, 2], near: .001, far: 100 }} gl={{ antialias: true, powerPreference: 'high-performance' }}>



        <color attach="background" args={['#151b18']} />



        <ambientLight intensity={1.5} /><directionalLight position={[2, 4, 4]} intensity={2} />



        <Controls bounds={bounds} fitSignal={fitSignal} navigation={navigation} viewDirection={viewDirection} hasGeometry={!!geometry} />



        {grid && <gridHelper args={[Math.max(1, diagonal * 2), 20, '#52664d', '#2b392d']} position={[center.x, result ? -ymax : -.5, center.z]} />}



        {axes && <axesHelper args={[Math.max(.1, diagonal * .35)]} position={[center.x, result ? -ymax : 0, center.z]} />}



        <group rotation={[Math.PI, 0, 0]}>



          {geometry && showCloud && <Cloud data={geometry.cloud} pointSize={pointSize} density={density} />}



          {geometry?.vertices && geometry.triangles && effectiveMode !== 'points' && <Surface data={geometry.vertices} indices={geometry.triangles} opacity={opacity} wireframe={wireframe} />}



          {geometry && normals && <Normals data={geometry.cloud} length={diagonal * .025} />}



        </group>



      </Canvas></ViewerBoundary>



      {!geometry && <div className="world-empty"><Box size={28} /><h3>{loading ? 'Loading reconstruction…' : 'Your object belongs here'}</h3><p>{error || 'Collect object images, then reconstruct the saved set. Aligned surfaces appear here.'}</p></div>}



      <div className="world-corner mono">METERS · Y UP DISPLAY</div>



      <div className="world-count mono">{geometry ? count.toLocaleString() + ' POINTS · ' + triangles.toLocaleString() + ' TRIANGLES' : 'NO GEOMETRY'}</div>



    </div>



    <div className="viewer-toolbar">



      <select aria-label="3D display mode" value={mode} onChange={e => setMode(e.target.value)}><option value="auto">Auto · surface first</option><option value="points">Point cloud</option><option value="mesh" disabled={!triangles}>Mesh</option><option value="both" disabled={!triangles}>Points + mesh</option></select>



      <button className="button secondary" onClick={() => setFitSignal(v => v + 1)}><Expand size={13} />Fit object</button>



      <button className="button secondary" onClick={() => setFitSignal(v => v + 1)} aria-label="Reset 3D camera"><RotateCcw size={13} /></button>



      <button className={'button secondary ' + (navigation === 'orbit' ? 'active' : '')} aria-pressed={navigation === 'orbit'} onClick={() => setNavigation('orbit')}>Orbit</button>



      <button className={'button secondary ' + (navigation === 'pan' ? 'active' : '')} aria-pressed={navigation === 'pan'} onClick={() => setNavigation('pan')}>Pan</button>



      <label><input type="checkbox" checked={grid} onChange={e => setGrid(e.target.checked)} />Grid</label><label><input type="checkbox" checked={axes} onChange={e => setAxes(e.target.checked)} />Axes</label>



    </div>



    <div className="view-directions" aria-label="Standard 3D views">{['front', 'back', 'top', 'bottom', 'left', 'right', 'angle'].map(direction => <button key={direction} className="button secondary" onClick={() => { setViewDirection(direction); setFitSignal(v => v + 1) }}>{direction}</button>)}</div>



    <p className="small muted viewer-hint">Drag to {navigation} · right drag / two fingers: pan · wheel: zoom. Front/up follow the first scan frame, not automatic semantic labels.</p>



    <details className="settings-details viewer-settings"><summary>3D appearance & controls</summary><div className="parameter-grid">



      <label className="field">Point size (m)<input type="number" min={.0005} max={.05} step={.0005} value={pointSize} onChange={e => setPointSize(Math.max(.0005, Math.min(.05, Number(e.target.value))))} /></label>



      <label className="field">Point density<select value={density} onChange={e => setDensity(Number(e.target.value))}>{[100, 50, 25, 10].map(n => <option key={n} value={n}>{n}%</option>)}</select></label>



      <label className="field">Mesh opacity<input type="range" min={.1} max={1} step={.05} value={opacity} onChange={e => setOpacity(Number(e.target.value))} /></label>



      <label className="checkbox-field"><input type="checkbox" checked={wireframe} onChange={e => setWireframe(e.target.checked)} />Wireframe</label><label className="checkbox-field"><input type="checkbox" checked={normals} onChange={e => setNormals(e.target.checked)} />Surface normals</label>



    </div><p className="small muted">Left drag: orbit · right drag: pan · wheel: zoom. Camera coordinates are rotated once into the Three.js display frame.</p></details>



    {result?.geometry_kind === 'visual_hull' && <p className="small muted">Outer-shape estimate · hidden concavities and unseen detail are not measured.</p>}

    {result && <div className="world-footer"><span>Estimated extent {result.dimensions_m.map(n => (n * 100).toFixed(1)).join(' × ')} cm</span><span>{result.calibration_source.replaceAll('_', ' ')}</span></div>}



  </section>



}











