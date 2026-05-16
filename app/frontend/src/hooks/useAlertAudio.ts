/**
 * Shared audio cue hook used by both the Live page and the inline
 * replay section of the Upload page.
 *
 * Browser autoplay policies disallow creating an AudioContext until
 * the first user gesture, so we don't construct one eagerly — instead
 * `unlock()` is called from a click / pointerdown handler the first
 * time the user interacts with the page.
 *
 * `chime(freq, ms)`        — one-shot tone (used on tier promotions).
 * `startAlertLoop()`       — repeating beep at ALERT_BEEP_INTERVAL_MS.
 * `stopAlertLoop()`        — silence the loop. Idempotent.
 *
 * The hook also cleans up the running interval when the consuming
 * component unmounts.
 */
import { useCallback, useEffect, useRef } from "react";

const ALERT_BEEP_INTERVAL_MS = 3000;

interface UseAlertAudio {
  unlock: () => void;
  chime: (freq: number, durationMs: number) => void;
  startAlertLoop: () => void;
  stopAlertLoop: () => void;
}

export function useAlertAudio(): UseAlertAudio {
  const ctxRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number | null>(null);

  const unlock = useCallback((): AudioContext | null => {
    if (ctxRef.current) return ctxRef.current;
    try {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      ctxRef.current = new Ctor();
      return ctxRef.current;
    } catch {
      return null;
    }
  }, []);

  const chime = useCallback((freq: number, durationMs: number) => {
    const ctx = unlock();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(
      0.0001, ctx.currentTime + durationMs / 1000,
    );
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + durationMs / 1000 + 0.05);
  }, [unlock]);

  const stopAlertLoop = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startAlertLoop = useCallback(() => {
    if (timerRef.current !== null) return;
    chime(660, 250);
    timerRef.current = window.setInterval(
      () => chime(660, 250),
      ALERT_BEEP_INTERVAL_MS,
    );
  }, [chime]);

  // Belt-and-braces: stop the loop when the consumer unmounts even if
  // they forgot to call stopAlertLoop themselves.
  useEffect(() => () => {
    if (timerRef.current !== null) clearInterval(timerRef.current);
  }, []);

  return {
    unlock: () => {
      unlock();
    },
    chime,
    startAlertLoop,
    stopAlertLoop,
  };
}
