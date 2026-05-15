/**
 * WebSocket client for /api/stream (T8/T9).
 *
 * Wraps the protocol so React components don't need to think about
 * frame types. The component receives typed callbacks (onTick, onEnd,
 * onError) and exposes pause/resume/setSpeed/stop methods.
 *
 * The WebSocket URL is derived from the current location so the same
 * code works under `vite dev` (5173 → proxied to 8000) and behind nginx
 * in docker (8080 → reverse-proxied to api:8000). NEVER hard-code the
 * port — we'd break one of the two environments.
 */

export type StreamStatus = "NORMAL" | "WATCH" | "WARNING" | "ALERT";

export interface TickFrame {
  type: "tick";
  idx: number;
  prob: number;
  status: StreamStatus;
  timestamp: string;
  in_anomaly_window: boolean;
  /** Per-channel reading at the window's last sample. Sent by the live
   *  endpoint; the CSV-replay stream omits it (the chart tooltip
   *  degrades gracefully). */
  sensors?: Record<string, number>;
}

export interface EndFrame {
  type: "end";
  summary: {
    total_windows: number;
    fault_windows: number;
    peak_idx: number;
    peak_prob: number;
    metrics: { precision: number; recall: number; f1: number } | null;
  };
}

export interface ErrorFrame {
  type: "error";
  code: string;
  message: string;
}

export interface StartConfig {
  fileId: string;
  modelId: string;
  threshold: number;
  speed: number;
}

export interface StreamHandlers {
  onTick: (tick: TickFrame) => void;
  onEnd: (end: EndFrame) => void;
  onError: (err: ErrorFrame) => void;
  onClose?: () => void;
}

export class StreamSession {
  private ws: WebSocket;

  constructor(cfg: StartConfig, handlers: StreamHandlers) {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/stream`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.ws.send(JSON.stringify({
        type: "start",
        file_id: cfg.fileId,
        model_id: cfg.modelId,
        threshold: cfg.threshold,
        speed: cfg.speed,
      }));
    };

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "tick") handlers.onTick(msg as TickFrame);
      else if (msg.type === "end") handlers.onEnd(msg as EndFrame);
      else if (msg.type === "error") handlers.onError(msg as ErrorFrame);
    };

    this.ws.onclose = () => handlers.onClose?.();
    this.ws.onerror = () =>
      handlers.onError({
        type: "error",
        code: "WS_TRANSPORT_ERROR",
        message: "WebSocket connection failed.",
      });
  }

  pause(): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "pause" }));
    }
  }

  resume(): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "resume" }));
    }
  }

  setSpeed(speed: number): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "set_speed", speed }));
    }
  }

  stop(): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        /* noop */
      }
    }
    this.close();
  }

  close(): void {
    if (
      this.ws.readyState === WebSocket.OPEN ||
      this.ws.readyState === WebSocket.CONNECTING
    ) {
      this.ws.close();
    }
  }
}
