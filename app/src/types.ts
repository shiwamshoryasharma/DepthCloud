export type CameraState = 'idle' | 'discovering' | 'starting' | 'running' | 'stopping' | 'error'

export interface CameraStatus {

  state: CameraState

  error: string | null

  generation: string

  camera_index: number | null

  requested: { index: number; width: number; height: number; fps: number; backend: string } | null

  actual: { width?: number; height?: number; backend?: string; driver_fps?: number; unsupported_requests?: string[] }

  current_buffer_size: number

  maximum_buffer_size: number

  buffer_fill: number

  buffer_bytes: number

  buffer_memory_limit: number

  buffer_evictions: number

  buffer_sequence_ids: number[]

  automatic_buffer: boolean

  frames_received: number

  frames_dropped: number

  capture_queue_drops: number

  preview_skips: number

  capture_fps: number

  processing_fps: number

  processing_latency_ms: number

  preview_latency_ms: number

  latest_sequence: number | null

  preview_options: PreviewOptions

}

export interface SystemStatus {

  device: { type: 'cpu' | 'cuda'; name: string; cuda_available: boolean; error: string | null }

  models: Record<string, { files_present: boolean; loaded: boolean; inference_enabled: boolean; bytes: number; loading?: boolean; error?: string | null; latency_ms?: number | null }>

  uptime_seconds: number

  python: string

  memory_mb: number

  stream_clients: number

  vram_allocated_mb?: number

  vram_reserved_mb?: number

}

export interface CameraDevice { index: number; label: string; backend: string; width: number; height: number }

export interface Activity { id: number; time: string; message: string; error: boolean }



export interface ROI { x: number; y: number; width: number; height: number }

export interface FrozenFrame { id: string; sequence_id: number; timestamp: number; width: number; height: number }

export interface PreviewOptions {

  mode: 'rgb' | 'vivid' | 'mono' | 'chrome' | 'pixelated' | 'pseudo_ir' | 'grayscale'

  saturation: number; contrast: number; brightness: number; block_size: number; mono_threshold: number; ir_palette: 'inferno' | 'turbo'

}

export interface Intrinsics { fx: number; fy: number; cx: number; cy: number; width: number; height: number; distortion: number[]; source: string; rms: number | null }

export interface Detection { label: string; score: number; box: number[] }

export interface Observation {

  id: string; label: string; sequence_id: number; timestamp: number; width: number; height: number

  roi: ROI; roi_pixels: number[]; selected: boolean; status: string; error: string | null

  result: {

    reference_only?: boolean; detections?: Detection[]; target?: Detection | null; warning?: string | null; focal_px?: number; point_count?: number

    depth_outliers_removed?: number; raw_foreground_pixels?: number

    intrinsics?: Intrinsics; depth_stats?: { minimum: number; maximum: number; median: number; valid_percent: number }

    registration?: { method: string; fitness: number; overlap: number; rmse: number; correspondences: number; accepted: boolean }

  }

}

export interface GeometryQuality {
  median_silhouette_iou: number; mean_reprojection_px: number; camera_span_deg: number
  metric_scale_relative_mad: number; colored_fraction: number
}

export interface ReconstructionResult {
  geometry_kind?: 'visual_hull'; method?: string; limitations?: string; quality?: GeometryQuality


  revision: number; points: number; vertices: number; triangles: number; accepted: string[]

  rejected: { id: string; label: string; reason: string }[]; mesh_error: string | null

  dimensions_m: number[]; bounds: number[][]; units: string; frame: string; calibration_source: string

}

export interface ReconstructionState {

  session_id: string; state: string; message: string; busy: boolean; progress: number; error: string | null

  active_view: string | null; views: Observation[]; result: ReconstructionResult | null; revision: number

  memory_bytes: number; memory_limit: number; max_views: number; elapsed_seconds: number

  capture?: CaptureScanStatus | null

  live?: LiveScanStatus | null

  calibration: { intrinsics: Intrinsics | null; samples: number; maximum_samples: number }

}

export interface PipelineOptions {

  threshold: number; require_detection: boolean; depth_min: number; depth_max: number; stride: number

  voxel_size: number; icp_distance: number; icp_iterations: number; min_fitness: number

  depth_cleanup: boolean; depth_outlier_strength: number

  normal_radius: number; outlier_neighbors: number; outlier_std: number; create_mesh: boolean; mesh_radius_factor: number

}





export interface LiveScanStatus {
  scan_mode: 'moving_camera' | 'rotating_object'
  camera_tracking: string
  camera_pose: number[][] | null
  registration?: { scene_inliers?: number; reprojection_px?: number; depth_scale?: number; relocalized?: boolean }


  reference_ids: string[]; reference_count: number; prepared_references: number
  matched_reference: { reference_id: string; label: string; score: number; box: number[]; method: string } | null
  preview_sequence: number | null
  processed: number; integrated: number; rejected: number; skipped: number

  last_sequence: number | null; tracking: string; last_error: string | null; surface_updates: number

  directions: string[]; orientation: string; processing_fps: number; voxel_size: number | null

  stopping: boolean; geometry_only: boolean

}


export interface CaptureScanStatus {
  reconstruction_quality?: GeometryQuality

  phase: string; message: string; active: boolean; collected: number; target: number
  sealed: boolean; disk_bytes: number; interval_seconds: number; processed?: number
  registered?: number; rejected?: number; skipped?: number; latest?: number; error?: string | null
  quality?: { stable: boolean; reason: string; sharpness: number; motion_px_s: number; hold_seconds: number }
  directions?: string[]; loop_closures?: number; stage_total?: number; reference_count?: number
}
