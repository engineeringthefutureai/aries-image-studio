import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface BatchCarouselProps {
  /** One tile per image, each with a stable key. */
  children: React.ReactElement[];
  selectedIndex: number;
  /** Scroll the selected tile into view when it changes (off: stay where the user is). */
  followSelection?: boolean;
  onSelect: (index: number) => void;
}

/** Edge fade, in px, shown on a side only while there is more to scroll that way. */
const FADE_PX = 28;

/**
 * A batch's tiles in one row at full tile height. Swipe, wheel, the arrows, the dots or
 * ←/→ (while focused) move through it; the selected tile is always scrolled into view.
 */
export const BatchCarousel: React.FC<BatchCarouselProps> = ({ children, selectedIndex, followSelection = true, onSelect }) => {
  const trackRef = useRef<HTMLDivElement>(null);
  const [canPrev, setCanPrev] = useState(false);
  const [canNext, setCanNext] = useState(false);
  const count = children.length;

  const updateEdges = useCallback(() => {
    const el = trackRef.current;
    if (!el) return;
    setCanPrev(el.scrollLeft > 1);
    setCanNext(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  }, []);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    updateEdges();
    const observer = new ResizeObserver(updateEdges);
    observer.observe(el);
    Array.from(el.children).forEach((child) => observer.observe(child));
    el.addEventListener('scroll', updateEdges, { passive: true });
    return () => {
      observer.disconnect();
      el.removeEventListener('scroll', updateEdges);
    };
  }, [updateEdges, count]);

  // Bring the selected tile fully into view without moving the page vertically.
  const scrollToIndex = useCallback((index: number, behavior: ScrollBehavior = 'smooth') => {
    const el = trackRef.current;
    const tile = el?.children[index] as HTMLElement | undefined;
    if (!el || !tile) return;
    const left = tile.offsetLeft - el.offsetLeft;
    const right = left + tile.offsetWidth;
    if (left < el.scrollLeft + FADE_PX) {
      el.scrollTo({ left: Math.max(0, left - FADE_PX), behavior });
    } else if (right > el.scrollLeft + el.clientWidth - FADE_PX) {
      el.scrollTo({ left: right - el.clientWidth + FADE_PX, behavior });
    }
  }, []);

  useEffect(() => {
    if (followSelection) scrollToIndex(selectedIndex);
  }, [selectedIndex, followSelection, scrollToIndex]);

  const go = (delta: number) => {
    const next = Math.min(count - 1, Math.max(0, selectedIndex + delta));
    if (next !== selectedIndex) onSelect(next);
    else scrollToIndex(next);
  };

  const page = (direction: 1 | -1) => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: direction * el.clientWidth * 0.8, behavior: 'smooth' });
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.target !== e.currentTarget) return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      go(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      go(1);
    }
  };

  const mask = `linear-gradient(to right, ${canPrev ? 'transparent' : 'black'} 0, black ${FADE_PX}px, black calc(100% - ${FADE_PX}px), ${
    canNext ? 'transparent' : 'black'
  } 100%)`;

  const arrow =
    'absolute top-1/2 -translate-y-1/2 z-10 grid place-items-center w-8 h-8 rounded-full bg-[#fcf9f2]/90 dark:bg-[#1a1a17]/90 border border-[#e5e2db] dark:border-[#2e2e2a] text-[#58423a] dark:text-[#d6cfc4] shadow-md backdrop-blur-sm transition-opacity hover:bg-white dark:hover:bg-[#232320] focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-amber-500';

  return (
    <div className="group/carousel relative w-full min-w-0" aria-roledescription="carousel" aria-label={`Batch of ${count} images`}>
      <div
        ref={trackRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        className="carousel-track flex gap-3.5 py-2.5 px-1.5 overflow-x-auto overscroll-x-contain snap-x snap-mandatory scroll-px-1.5 rounded-xl focus-visible:outline-2 focus-visible:outline-amber-500/60"
        style={{ maskImage: mask, WebkitMaskImage: mask }}
      >
        {children.map((child, i) => (
          <div
            key={child.key ?? i}
            className="snap-start shrink-0"
            role="group"
            aria-roledescription="slide"
            aria-label={`Image ${i + 1} of ${count}`}
          >
            {child}
          </div>
        ))}
      </div>

      {canPrev && (
        <button
          type="button"
          onClick={() => page(-1)}
          className={`${arrow} left-1 sm:opacity-0 sm:group-hover/carousel:opacity-100`}
          aria-label="Scroll left"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      )}
      {canNext && (
        <button
          type="button"
          onClick={() => page(1)}
          className={`${arrow} right-1 sm:opacity-0 sm:group-hover/carousel:opacity-100`}
          aria-label="Scroll right"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      )}

      {/* The dots' row always takes its space (hidden while every image fits): it appearing
          after the first measurement shifted the page, and Back restored the wrong position. */}
      {count > 1 && (
        <div
          className={`flex justify-center items-center gap-1.5 pt-1 ${canPrev || canNext ? '' : 'invisible'}`}
          aria-hidden={!(canPrev || canNext)}
        >
          {children.map((child, i) => (
            <button
              key={child.key ?? i}
              type="button"
              onClick={() => (i === selectedIndex ? scrollToIndex(i) : onSelect(i))}
              className="p-1 -m-0.5"
              aria-label={`Show image ${i + 1}`}
              aria-current={i === selectedIndex}
            >
              <span
                className={`block h-1.5 rounded-full transition-all ${
                  i === selectedIndex ? 'w-4 bg-amber-500' : 'w-1.5 bg-[#d6cfc4] dark:bg-[#3a3a35] hover:bg-[#b5a89e]'
                }`}
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
