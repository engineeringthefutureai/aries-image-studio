import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';

interface ZoomableImageProps {
  src: string;
  /** Small copy for the navigator; falls back to `src`. */
  thumbSrc?: string;
  alt: string;
  /** Image size in pixels; 100 % shows one image pixel per device pixel. */
  width: number;
  height: number;
  /** Reports whether the image is larger than the stage, so ←/→ pan instead of navigating. */
  onZoomedChange?: (zoomed: boolean) => void;
}

/** Zoom levels (1 = actual size) for +/− and the buttons; wheel and pinch zoom continuously. */
const STEPS = [0.1, 0.125, 0.167, 0.25, 0.333, 0.5, 0.667, 1, 1.5, 2, 3, 4, 6, 8];
const MAX_ZOOM = 8;
/** Space kept around the image when it is fitted. */
const FIT_PAD = 24;
/** Arrow-key pan distance in CSS pixels. */
const KEY_PAN = 120;
/** Navigator width: at most this, and at most a quarter of the stage. */
const NAV_WIDTH = 168;
/** Movement (CSS px) after which a press is a drag rather than a click. */
const CLICK_SLOP = 5;

interface View {
  /** null = fit to the stage (follows resizes). */
  scale: number | null;
  /** Image centre offset from the stage centre, CSS pixels. */
  x: number;
  y: number;
  /** Animate the change (buttons, keys); wheel, pinch and drags follow the pointer directly. */
  smooth: boolean;
}

/** CSS pixels per image pixel at 100 %: one device pixel, as in photo editors. */
const devicePixel = () => 1 / (window.devicePixelRatio || 1);

/**
 * The viewer's image: fit to the stage, zoom to true pixel sizes (click for 100 % and back,
 * wheel or pinch at the pointer, +/−), pan by dragging, arrow keys or the navigator.
 */
export const ZoomableImage: React.FC<ZoomableImageProps> = ({
  src,
  thumbSrc,
  alt,
  width,
  height,
  onZoomedChange,
}) => {
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View>({ scale: null, x: 0, y: 0, smooth: true });
  const [dragging, setDragging] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ dist: number; scale: number } | null>(null);
  /** The current press: where it started, and whether it became a drag or a pinch. */
  const press = useRef<{ x: number; y: number; gesture: boolean } | null>(null);

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStage({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const fitScale =
    stage.w > 0 && stage.h > 0
      ? Math.min((stage.w - 2 * FIT_PAD) / width, (stage.h - 2 * FIT_PAD) / height)
      : 0;
  const unit = devicePixel();
  const maxScale = MAX_ZOOM * unit;
  const minScale = Math.min(fitScale, unit);
  const scale = view.scale ?? fitScale;
  /** Zoom as shown to the user: 1 = actual size. */
  const zoom = scale / unit;
  const imgW = width * scale;
  const imgH = height * scale;
  const zoomed = imgW > stage.w + 1 || imgH > stage.h + 1;

  useEffect(() => {
    onZoomedChange?.(zoomed);
  }, [zoomed, onZoomedChange]);

  /** Keep the image covering the stage on each axis it overflows; centre it otherwise. */
  const clamp = useCallback(
    (s: number, x: number, y: number, smooth: boolean): View => {
      const maxX = Math.max(0, (width * s - stage.w) / 2);
      const maxY = Math.max(0, (height * s - stage.h) / 2);
      return { scale: s, x: Math.max(-maxX, Math.min(maxX, x)), y: Math.max(-maxY, Math.min(maxY, y)), smooth };
    },
    [width, height, stage.w, stage.h]
  );

  /** Zoom to `next`, keeping the image point under (px, py) (stage-centre coordinates) still. */
  const zoomTo = useCallback(
    (next: number, px = 0, py = 0, smooth = true) => {
      setView((v) => {
        const cur = v.scale ?? fitScale;
        const s = Math.max(minScale, Math.min(maxScale, next));
        if (Math.abs(s - fitScale) < 1e-4) return { scale: null, x: 0, y: 0, smooth };
        const k = s / cur;
        return clamp(s, px - (px - v.x) * k, py - (py - v.y) * k, smooth);
      });
    },
    [clamp, fitScale, minScale, maxScale]
  );

  const step = useCallback(
    (dir: 1 | -1) => {
      const levels = [...STEPS.map((z) => z * unit), fitScale].sort((a, b) => a - b);
      const next = dir > 0 ? levels.find((l) => l > scale * 1.01) : [...levels].reverse().find((l) => l < scale * 0.99);
      zoomTo(next ?? (dir > 0 ? maxScale : minScale));
    },
    [fitScale, minScale, maxScale, unit, scale, zoomTo]
  );

  const fit = useCallback(() => setView({ scale: null, x: 0, y: 0, smooth: true }), []);
  const pan = useCallback(
    (dx: number, dy: number, smooth = false) =>
      setView((v) => (v.scale == null ? v : clamp(v.scale, v.x + dx, v.y + dy, smooth))),
    [clamp]
  );

  // Keyboard: +/− step, 0 fits, 1 is 100 %, arrows pan while zoomed in.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '+' || e.key === '=') step(1);
      else if (e.key === '-' || e.key === '_') step(-1);
      else if (e.key === '0') fit();
      else if (e.key === '1') zoomTo(unit);
      else if (zoomed && e.key.startsWith('Arrow')) {
        e.preventDefault();
        const d = e.shiftKey ? KEY_PAN * 4 : KEY_PAN;
        if (e.key === 'ArrowLeft') pan(d, 0, true);
        else if (e.key === 'ArrowRight') pan(-d, 0, true);
        else if (e.key === 'ArrowUp') pan(0, d, true);
        else if (e.key === 'ArrowDown') pan(0, -d, true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, fit, zoomTo, pan, zoomed, unit]);

  // Wheel (and trackpad pinch, which arrives as ctrl+wheel) zooms at the pointer.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      zoomTo(scale * factor, e.clientX - r.left - r.width / 2, e.clientY - r.top - r.height / 2, false);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [scale, zoomTo]);

  const toStage = (clientX: number, clientY: number) => {
    const r = stageRef.current!.getBoundingClientRect();
    return { x: clientX - r.left - r.width / 2, y: clientY - r.top - r.height / 2 };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest('button, [data-navigator]')) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) press.current = { x: e.clientX, y: e.clientY, gesture: false };
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      pinch.current = { dist: Math.hypot(a.x - b.x, a.y - b.y), scale };
      if (press.current) press.current.gesture = true;
    }
    setDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    const cur = { x: e.clientX, y: e.clientY };
    pointers.current.set(e.pointerId, cur);
    if (press.current && Math.hypot(cur.x - press.current.x, cur.y - press.current.y) > CLICK_SLOP) {
      press.current.gesture = true;
    }
    if (pointers.current.size === 2 && pinch.current) {
      const [a, b] = [...pointers.current.values()];
      const mid = toStage((a.x + b.x) / 2, (a.y + b.y) / 2);
      zoomTo((pinch.current.scale * Math.hypot(a.x - b.x, a.y - b.y)) / pinch.current.dist, mid.x, mid.y, false);
    } else if (pointers.current.size === 1) {
      pan(cur.x - prev.x, cur.y - prev.y);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!pointers.current.delete(e.pointerId)) return;
    if (pointers.current.size < 2) pinch.current = null;
    if (pointers.current.size > 0) return;
    setDragging(false);
    // A click (no drag, no pinch) zooms in to 100 % at the pointer, or back out to Fit.
    const wasClick = e.type === 'pointerup' && press.current && !press.current.gesture;
    press.current = null;
    if (!wasClick) return;
    if (view.scale != null && scale > fitScale + 1e-4) {
      fit();
    } else {
      const p = toStage(e.clientX, e.clientY);
      // A small image whose 100 % fits on screen zooms to twice its fitted size instead.
      zoomTo(unit > fitScale * 1.01 ? unit : fitScale * 2, p.x, p.y);
    }
  };

  // Navigator: the whole image with the visible part outlined; click or drag to move it.
  const navW = Math.min(NAV_WIDTH, stage.w / 4, (stage.h / 4) * (width / height));
  const navH = (navW * height) / width;
  const navK = navW / imgW;
  const visW = Math.min(stage.w, imgW) * navK;
  const visH = Math.min(stage.h, imgH) * navK;
  const visX = (imgW / 2 - view.x - Math.min(stage.w, imgW) / 2) * navK;
  const visY = (imgH / 2 - view.y - Math.min(stage.h, imgH) / 2) * navK;

  const navMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.type === 'pointermove' && !e.currentTarget.hasPointerCapture(e.pointerId)) return;
    const r = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width - 0.5;
    const fy = (e.clientY - r.top) / r.height - 0.5;
    setView((v) => (v.scale == null ? v : clamp(v.scale, -fx * width * v.scale, -fy * height * v.scale, false)));
  };

  const isOneToOne = Math.abs(zoom - 1) < 0.005;
  const btn = 'p-1.5 rounded-full hover:bg-white/15 text-white/80 hover:text-white transition-colors disabled:opacity-30 disabled:hover:bg-transparent';
  const pill = (active: boolean) =>
    `flex items-center gap-1 px-2 py-1 rounded-full text-[11px] transition-colors ${
      active ? 'bg-white/20 text-white' : 'hover:bg-white/15 text-white/80 hover:text-white'
    }`;

  return (
    <div
      ref={stageRef}
      className={`absolute inset-0 overflow-hidden touch-none ${
        zoomed ? (dragging ? 'cursor-grabbing' : 'cursor-grab') : view.scale != null ? 'cursor-zoom-out' : 'cursor-zoom-in'
      }`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {fitScale > 0 &&
        // The thumbnail shows at once (usually cached from the grid) until the full PNG loads.
        [...(thumbSrc && !loaded ? [thumbSrc] : []), src].map((url) => (
          <img
            key={url}
            src={url}
            alt={url === src ? alt : ''}
            draggable={false}
            onLoad={url === src ? () => setLoaded(true) : undefined}
            className={`absolute left-1/2 top-1/2 max-w-none rounded-xl ${url === src ? 'shadow-2xl' : ''} ${
              dragging || !view.smooth ? '' : 'transition-[width,height,transform] duration-150 ease-out motion-reduce:transition-none'
            }`}
            style={{
              width: imgW,
              height: imgH,
              transform: `translate(calc(-50% + ${view.x}px), calc(-50% + ${view.y}px))`,
              imageRendering: zoom >= 2 && url === src ? 'pixelated' : 'auto',
            }}
          />
        ))}

      {zoomed && (
        <div
          data-navigator
          className="absolute top-3 left-3 sm:top-4 sm:left-4 z-20 rounded-lg overflow-hidden border border-white/20 shadow-xl bg-black/60 cursor-crosshair touch-none"
          style={{ width: navW, height: navH }}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            navMove(e);
          }}
          onPointerMove={navMove}
          title="Navigator: click or drag to move"
        >
          <img src={thumbSrc || src} alt="" draggable={false} className="w-full h-full object-cover opacity-80" />
          <div
            className="absolute border-2 border-amber-400 shadow-[0_0_0_9999px_rgba(0,0,0,0.45)] pointer-events-none"
            style={{ left: visX, top: visY, width: visW, height: visH }}
          />
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 px-2.5 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-white/15 shadow-xl text-xs font-mono">
        <button onClick={() => step(-1)} disabled={scale <= minScale + 1e-4} className={btn} title="Zoom out (−)">
          <ZoomOut className="w-4 h-4" />
        </button>
        <span className="w-12 text-center text-[11px] font-mono text-white/90" title="Of the image's actual size">
          {zoom < 0.1 ? (zoom * 100).toFixed(1) : Math.round(zoom * 100)}%
        </span>
        <button onClick={() => step(1)} disabled={scale >= maxScale - 1e-4} className={btn} title="Zoom in (+)">
          <ZoomIn className="w-4 h-4" />
        </button>
        <div className="w-px h-3.5 bg-white/20 mx-1" />
        <button onClick={fit} className={pill(view.scale == null)} title="Fit to screen (0)">
          <Maximize className="w-3.5 h-3.5" />
          <span>Fit</span>
        </button>
        <button onClick={() => zoomTo(unit)} className={pill(isOneToOne)} title="Actual size, one image pixel per screen pixel (1, or click the image)">
          <span>1:1</span>
        </button>
      </div>
    </div>
  );
};
