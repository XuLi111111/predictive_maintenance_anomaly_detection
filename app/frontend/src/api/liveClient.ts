/**
 * WebSocket client for /api/live/stream.
 *
 * Unlike the CSV-replay stream (streamClient.ts) which has a finite
 * start → tick → end lifecycle, the live stream is open-ended: ticks
 * keep coming as long as the simulator (or, in production, the SCADA
 * bridge) keeps pushing samples to /api/live/ingest.
 *
 * If the connection drops we transparently reconnect with backoff so
 * the operator's dashboard self-heals when the network blips.
 */

export type LiveStreamStatus = "NORMAL" | "WATCH" | "WARNING" | "ALERT";

export interface LiveTick {
  type: "tick";
  idx: number;
  prob: number;
  status: LiveStreamStatus;
  timestamp: string;
  in_anomaly_window: boolean;
  /** Per-channel reading at the window's last sample. The 8 SKAB
   *  channels are keyed by name (e.g. "Accelerometer1RMS"). */
  sensors?: Record<string, number>;
}

export interface LiveHello {
  type: "hello";
  samples_in_buffer: number;
  window_size: number;
  current_status: LiveStreamStatus;
}

export interface LiveConfigFrame {
  type: "config";
  model_id: string;
  threshold: number;
}

export type QualityCode =
  | "FROZEN_SENSOR"
  | "FROZEN_CLEARED"
  | "OUT_OF_RANGE"
  | "OUT_OF_RANGE_CLEARED"
  | "UNEVEN_SAMPLING"
  | "EVEN_SAMPLING_RESTORED";

export interface LiveQualityFrame {
  type: "quality";
  code: QualityCode;
  channel: string | null;
  severity: "warning" | "info";
  message: string;
}

/** Connection state surfaced to the UI badge. */
export type LiveConnState =
  | "connecting"   // socket opening
  | "waiting"     // socket open but no tick yet (warming up / simulator not running)
  | "live"        // ticks flowing
  | "disconnected"; // socket closed; will retry

export interface LiveHandlers {
  onTick: (tick: LiveTick) => void;
  onHello?: (hello: LiveHello) => void;
  onConfig?: (cfg: LiveConfigFrame) => void;
  onQuality?: (frame: LiveQualityFrame) => void;
  onState: (state: LiveConnState) => void;
}

export async function postLiveConfig(
  patch: { model_id?: string; threshold?: number },
): Promise<{ ok: boolean; model_id: string; threshold: number }> {
  const res = await fetch("/api/live/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = (j as { detail?: string }).detail ?? msg;
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

const RECONNECT_DELAY_MS = 2500;

export class LiveSession {
  private ws: WebSocket | null = null;
  private retryTimer: number | null = null;
  private closed = false;
  private state: LiveConnState = "connecting";

  constructor(private readonly handlers: LiveHandlers) {
    this.connect();
  }

  close(): void {
    this.closed = true;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* noop */
      }
      this.ws = null;
    }
  }

  private connect(): void {
    if (this.closed) return;
    this.setState("connecting");

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/live/stream`;

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this.scheduleRetry();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.setState("waiting");
    };

    ws.onmessage = (event) => {
      let msg: unknown;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      const m = msg as { type?: string };
      if (m.type === "hello") {
        this.handlers.onHello?.(msg as LiveHello);
      } else if (m.type === "tick") {
        this.setState("live");
        this.handlers.onTick(msg as LiveTick);
      } else if (m.type === "config") {
        this.handlers.onConfig?.(msg as LiveConfigFrame);
      } else if (m.type === "quality") {
        this.handlers.onQuality?.(msg as LiveQualityFrame);
      }
    };

    ws.onclose = () => {
      if (this.closed) return;
      this.setState("disconnected");
      this.scheduleRetry();
    };

    ws.onerror = () => {
      // onclose will fire after onerror; let it drive the retry.
    };
  }

  private scheduleRetry(): void {
    if (this.closed || this.retryTimer !== null) return;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, RECONNECT_DELAY_MS);
  }

  private setState(next: LiveConnState): void {
    if (this.state === next) return;
    this.state = next;
    this.handlers.onState(next);
  }
}
