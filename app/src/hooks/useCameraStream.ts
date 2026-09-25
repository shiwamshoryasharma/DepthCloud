import { useCallback, useEffect, useRef, useState } from 'react'
import type { Activity, CameraStatus, SystemStatus, ReconstructionState } from '../types'

export function useCameraStream() {
  const [connection, setConnection] = useState<'connecting' | 'online' | 'offline'>('connecting')
  const [camera, setCamera] = useState<CameraStatus | null>(null)
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [reconstruction, setReconstruction] = useState<ReconstructionState | null>(null)
  const [image, setImage] = useState<string | null>(null)
  const [activity, setActivity] = useState<Activity[]>([])
  const [fpsHistory, setFpsHistory] = useState<number[]>([])
  const socket = useRef<WebSocket | null>(null)
  const objectUrl = useRef<string | null>(null)
  const state = useRef<string | null>(null)
  const activityId = useRef(0)

  const record = useCallback((message: string, error = false) => {
    const entry = { id: ++activityId.current, time: new Date().toLocaleTimeString([], { hour12: false }), message, error }
    setActivity(previous => [entry, ...previous].slice(0, 8))
  }, [])

  const acknowledge = useCallback(() => {
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send('ack')
  }, [])

  useEffect(() => {
    let disposed = false
    let retry: ReturnType<typeof setTimeout>
    let attempt = 0
    const clearImage = () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = null
      setImage(null)
    }
    const connect = () => {
      if (disposed) return
      setConnection('connecting')
      const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/camera')
      socket.current = ws
      ws.onopen = () => {
        if (disposed) { ws.close(); return }
        attempt = 0
        setConnection('online')
        record('Connected to DepthCloud')
      }
      ws.onmessage = event => {
        if (disposed) return
        if (event.data instanceof Blob) {
          if (state.current !== 'running') { acknowledge(); return }
          if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
          objectUrl.current = URL.createObjectURL(event.data)
          setImage(objectUrl.current)
          return
        }
        let payload
        try { payload = JSON.parse(event.data) } catch { return }
        if (payload.type !== 'status') return
        const next = payload.camera as CameraStatus
        setCamera(next)
        setSystem(payload.system as SystemStatus)
        setReconstruction(payload.reconstruction as ReconstructionState)
        if (next.state !== state.current) {
          const labels: Record<string, string> = {
            idle: 'Camera idle · transient frames cleared',
            discovering: 'Scanning local camera devices',
            starting: 'Opening camera connection',
            running: 'Continuous capture is running',
            stopping: 'Stopping capture',
            error: next.error || 'Camera failed',
          }
          record(labels[next.state] || next.state, next.state === 'error')
          state.current = next.state
        }
        if (next.state !== 'running') {
          clearImage()
          acknowledge()
          setFpsHistory([])
        } else {
          setFpsHistory(previous => [...previous, next.capture_fps].slice(-60))
        }
      }
      ws.onerror = () => ws.close()
      ws.onclose = () => {
        if (disposed) return
        setConnection('offline')
        state.current = null
        clearImage()
        setCamera(null)
        setFpsHistory([])
        record('Stream disconnected · reconnecting', true)
        retry = setTimeout(connect, Math.min(5000, 500 * 2 ** attempt++))
      }
    }
    retry = setTimeout(connect, 0)
    return () => {
      disposed = true
      clearTimeout(retry)
      socket.current?.close()
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = null
    }
  }, [acknowledge, record])

  return { connection, camera, system, reconstruction, image, activity, fpsHistory, acknowledge, record }
}


