import React, { useEffect, useRef } from 'react';
import { Loader2 } from 'lucide-react';

interface LoadMoreProps {
  hasMore: boolean;
  loading: boolean;
  onLoadMore: () => void;
}

/**
 * The end of an infinitely scrolling list: asks for the next page well before it scrolls into
 * view. The observer is renewed after every page, so a page too short to fill the screen still
 * brings the next one.
 */
export const LoadMore: React.FC<LoadMoreProps> = ({ hasMore, loading, onLoadMore }) => {
  const ref = useRef<HTMLDivElement>(null);
  const onLoadMoreRef = useRef(onLoadMore);
  useEffect(() => {
    onLoadMoreRef.current = onLoadMore;
  }, [onLoadMore]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !hasMore || loading) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) onLoadMoreRef.current();
      },
      { rootMargin: '1500px 0px' },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loading]);

  if (!hasMore) return null;
  return (
    <div ref={ref} className="flex justify-center py-8 text-[#8c7169] dark:text-[#a09088]" aria-live="polite">
      {loading && <Loader2 className="w-5 h-5 animate-spin" aria-label="Loading more" />}
    </div>
  );
};
