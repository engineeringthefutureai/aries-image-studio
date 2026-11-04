import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  type Generation,
  type AspectRatio,
  type SearchHit,
  type SearchStrength,
  ASPECT_RATIOS,
} from '../types';
import { aspectOf, type GalleryFacets } from '../services/api';
import type { GalleryFilters } from '../utils/router';
import { imageHref, onLinkClick } from '../utils/router';
import { LoadMore } from './LoadMore';
import { formatDuration, generationSeconds } from '../utils/timings';
import {
  Globe,
  Heart,
  Dices,
  Maximize2,
  Download,
  Search,
  SlidersHorizontal,
  X,
  Image as ImageIcon,
  Trash2,
  Eye,
  Type,
  TextQuote,
  Hash,
  ScanSearch,
} from 'lucide-react';
import { PageLoader } from './PageLoader';

interface GalleryProps {
  /** Finished images matching `filters`, newest first (Studio filters them, a page at a time). */
  generations: Generation[];
  filters: GalleryFilters;
  /** New filters: the page's URL changes with them (`replace` while typing a search). */
  onFiltersChange: (filters: GalleryFilters, replace?: boolean) => void;
  /** Counts over all finished images, for the filter pills. */
  facets: GalleryFacets | null;
  /** Absent for a demo visitor. */
  onToggleFavorite?: (id: string) => void;
  /** Show this image in the demo or stop (the owner). */
  onToggleDemo?: (generation: Generation) => void;
  onVary: (generation: Generation) => void;
  onDelete?: (id: string) => void;
  /** The first page has not arrived yet: show a loader, not the empty state. */
  loading?: boolean;
  hasMore?: boolean;
  loadingMore?: boolean;
  onLoadMore?: () => void;
}

/** Column width of the masonry grid (max-w-7xl, px-4/sm:px-6, 1–4 columns, gap-4), for `sizes`. */
const GALLERY_TILE_SIZES =
  '(min-width: 1280px) 296px, (min-width: 1024px) calc((100vw - 80px) / 3), (min-width: 640px) calc((100vw - 64px) / 2), calc(100vw - 32px)';

const noop = () => {};

const STRENGTH_LABEL: Record<SearchStrength, string> = {
  strong: 'Strong look match',
  possible: 'Possible look match',
  weak: 'Weak look match',
};
const STRENGTH_BARS: Record<SearchStrength, number> = { strong: 3, possible: 2, weak: 1 };

/** How a search found an image, in words (docs/SEARCH.md): "Strong look match · Exact phrase
 * in your prompt". */
function describeHit(hit: SearchHit): string {
  const parts: string[] = [];
  if (hit.match !== 'text') parts.push(hit.strength ? STRENGTH_LABEL[hit.strength] : 'Found by what it shows');
  if (hit.match !== 'semantic') {
    if (hit.words === 'seed') parts.push('Seed matches');
    else if (hit.words === 'enhanced') parts.push(`${hit.phrase ? 'Exact phrase' : 'Words'} only in the enhanced prompt`);
    else parts.push(`${hit.phrase ? 'Exact phrase' : 'Words'} in your prompt`);
  }
  return parts.join(' · ');
}

/** Found only by a faint resemblance: Studio sends these last, under "Weaker matches". */
const isWeak = (g: Generation) => g.search?.match === 'semantic' && g.search.strength === 'weak';

/** Signal bars: how closely an image looks like the search. */
const StrengthBars: React.FC<{ strength: SearchStrength }> = ({ strength }) => (
  <span className="flex items-end gap-px h-3" aria-hidden="true">
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className={`w-[3px] rounded-[1px] ${i < STRENGTH_BARS[strength] ? 'bg-white' : 'bg-white/30'}`}
        style={{ height: `${6 + i * 3}px` }}
      />
    ))}
  </span>
);

/** The words' icon: an exact phrase, words, or a seed; dimmed when only the enhanced prompt
 * has them. */
const WordsIcon: React.FC<{ hit: SearchHit }> = ({ hit }) => {
  const Icon = hit.words === 'seed' ? Hash : hit.phrase ? TextQuote : Type;
  return <Icon className={`w-3 h-3 ${hit.words === 'enhanced' ? 'opacity-50' : ''}`} />;
};

export const Gallery: React.FC<GalleryProps> = ({
  generations,
  filters,
  onFiltersChange,
  facets,
  onToggleFavorite,
  onToggleDemo,
  onVary,
  onDelete,
  loading = false,
  hasMore = false,
  loadingMore = false,
  onLoadMore = noop,
}) => {
  const onlyFavorites = filters.favorites;
  const selectedRatio = filters.aspect || 'all';
  // The search box types freely; the URL (and Studio's query) follow after a pause.
  const [searchQuery, setSearchQuery] = useState(filters.q);
  const searchBox = useRef<HTMLInputElement>(null);
  // Back/Forward (or a reset) changes the URL's query: the box follows it. The URL holds the
  // query trimmed: that one is the box's own, and keeps the space just typed.
  const [urlQuery, setUrlQuery] = useState(filters.q);
  if (urlQuery !== filters.q) {
    setUrlQuery(filters.q);
    if (filters.q !== searchQuery.trim()) setSearchQuery(filters.q);
  }
  useEffect(() => {
    if (searchQuery.trim() === filters.q) return;
    // A search replaces "more like this".
    const timer = setTimeout(() => onFiltersChange({ ...filters, q: searchQuery, like: '' }, true), 300);
    return () => clearTimeout(timer);
  }, [searchQuery, filters, onFiltersChange]);

  // Counts per aspect label, from Studio's per-size counts (½ and ¼ sizes share a label).
  const aspectCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of facets?.sizes || []) {
      const label = aspectOf(s.width, s.height);
      counts.set(label, (counts.get(label) || 0) + s.count);
    }
    return counts;
  }, [facets]);
  const totalCount = facets?.total ?? generations.length;
  const favoriteCount = facets?.favorites ?? generations.filter((g) => g.favorite).length;

  const hasActiveFilters =
    onlyFavorites || filters.demo || selectedRatio !== 'all' || filters.q.trim() !== '' || filters.like !== '';
  // Every result of "more like this" is found by its look: no badges.
  const showMatch = filters.like === '';
  // The tile whose badge was tapped open (touch screens have no hover to explain it).
  const [openBadge, setOpenBadge] = useState<string | null>(null);
  const strongerResults = useMemo(() => generations.filter((g) => !isWeak(g)), [generations]);
  const weakerResults = useMemo(() => generations.filter(isWeak), [generations]);

  const resetFilters = () => {
    setSearchQuery('');
    onFiltersChange({ favorites: false, aspect: '', q: '', like: '', demo: false });
  };

  // Only the first load replaces the page: later ones (a search, a filter) keep the header
  // mounted, or the search box would lose focus (and a phone its keyboard) at each search.
  const [shown, setShown] = useState(false);
  const waiting = loading && generations.length === 0;
  if (!shown && !waiting) setShown(true);
  if (waiting && !shown) return <PageLoader />;

  if (!loading && !hasActiveFilters && generations.length === 0 && totalCount === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-lg mx-auto text-center space-y-5">
        <div className="w-14 h-14 rounded-full bg-[#f1eee7] dark:bg-[#20201d] flex items-center justify-center text-[#8c7169] dark:text-[#a09088] shadow-xs">
          <ImageIcon className="w-7 h-7" />
        </div>
        <div>
          <h2 className="font-['Plus_Jakarta_Sans'] font-bold text-xl text-[#1c1c18] dark:text-[#f3f0e9]">
            Gallery is empty
          </h2>
          <p className="text-sm text-[#58423a] dark:text-[#a09088] mt-1.5 leading-relaxed">
            Images generated in the timeline will appear here automatically. Type a prompt in the composer below to begin.
          </p>
        </div>
      </div>
    );
  }

  const renderTile = (gen: Generation) => {
    const aspectConfig = ASPECT_RATIOS[gen.params.aspect] || ASPECT_RATIOS['1:1'];
    const aspectRatioStyle = `${aspectConfig.ratioWidth} / ${aspectConfig.ratioHeight}`;
    const href = imageHref(gen.id);
    const hit = showMatch ? gen.search : undefined;

    return (
      <div
        key={gen.id}
        className="break-inside-avoid group relative rounded-2xl overflow-hidden bg-[#f1eee7] dark:bg-[#20201d] border border-[#e5e2db]/60 dark:border-[#2a2a26] shadow-xs hover:shadow-lg transition-all duration-300 cursor-pointer"
      >
        {/* Media Container */}
        <div className="relative w-full overflow-hidden" style={{ aspectRatio: aspectRatioStyle }}>
          <img
            src={gen.thumbUrl || gen.imageUrl}
            srcSet={gen.thumbSrcSet}
            sizes={GALLERY_TILE_SIZES}
            alt={gen.prompt}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover transition-transform duration-500 ease-out group-hover:scale-103"
          />

          {/* The tile opens the viewer: a real link (new tab, copy), under the action buttons */}
          <a
            href={href}
            onClick={(e) => onLinkClick(e, href, { from: 'gallery' })}
            className="absolute inset-0 z-[5]"
            aria-label={`Open: ${gen.prompt}`}
          />

          {/* Top Badges */}
          <div className="pointer-events-none absolute top-2.5 right-2.5 z-20 group-hover:opacity-0 transition-opacity flex items-center gap-1 bg-black/50 backdrop-blur-md px-2 py-0.5 rounded-full text-[10px] font-mono text-white/90">
            <span>{gen.params.aspect}</span>
          </div>

          {/* How the search found it; a tap spells it out (with a mouse, the hover overlay does) */}
          {hit && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setOpenBadge((open) => (open === gen.id ? null : gen.id));
              }}
              className="absolute bottom-2.5 left-2.5 z-20 group-hover:opacity-0 group-hover:pointer-events-none transition-opacity flex items-center gap-1 bg-black/50 backdrop-blur-md px-1.5 py-1 rounded-full text-white/90 text-[10px] leading-3"
              title={describeHit(hit)}
              aria-label={describeHit(hit)}
              aria-expanded={openBadge === gen.id}
            >
              {hit.match !== 'text' && (
                <>
                  <Eye className="w-3 h-3" />
                  {hit.strength && <StrengthBars strength={hit.strength} />}
                </>
              )}
              {hit.match !== 'semantic' && <WordsIcon hit={hit} />}
              {openBadge === gen.id && <span className="pl-0.5 pr-0.5">{describeHit(hit)}</span>}
            </button>
          )}

          {(gen.favorite || (gen.demo && onToggleDemo)) && (
            <div className="pointer-events-none absolute top-2.5 left-2.5 z-20 flex gap-1">
              {gen.favorite && (
                <div className="w-6 h-6 rounded-full bg-rose-500 text-white flex items-center justify-center shadow-xs">
                  <Heart className="w-3.5 h-3.5 fill-current" />
                </div>
              )}
              {gen.demo && onToggleDemo && (
                <div
                  className="w-6 h-6 rounded-full bg-sky-600 text-white flex items-center justify-center shadow-xs"
                  title="Shown publicly in the demo"
                >
                  <Globe className="w-3.5 h-3.5" />
                </div>
              )}
            </div>
          )}

          {/* Hover Scrim & Quick Action HUD */}
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-3.5 z-10">
            {/* Top Buttons */}
            <div className="pointer-events-auto flex items-center justify-end gap-1.5">
              {onToggleFavorite && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleFavorite(gen.id);
                  }}
                  className={`w-8 h-8 rounded-full backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110 ${
                    gen.favorite ? 'bg-rose-500 text-white' : 'bg-black/50 text-white hover:bg-black/70'
                  }`}
                  title="Favorite"
                >
                  <Heart className={`w-4 h-4 ${gen.favorite ? 'fill-current' : ''}`} />
                </button>
              )}
              {onToggleDemo && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleDemo(gen);
                  }}
                  className={`w-8 h-8 rounded-full backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110 ${
                    gen.demo ? 'bg-sky-600 text-white' : 'bg-black/50 text-white hover:bg-black/70'
                  }`}
                  title={gen.demo ? 'Shown publicly in the demo: stop showing it' : 'Show publicly in the demo'}
                  aria-pressed={!!gen.demo}
                >
                  <Globe className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onVary(gen);
                }}
                className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Vary: fill the composer from this image with a new seed"
              >
                <Dices className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onFiltersChange({ ...filters, q: '', like: gen.id });
                }}
                className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="More like this"
              >
                <ScanSearch className="w-4 h-4" />
              </button>
              <a
                href={href}
                onClick={(e) => onLinkClick(e, href, { from: 'gallery' })}
                className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Open in the viewer"
              >
                <Maximize2 className="w-4 h-4" />
              </a>
              <a
                href={gen.imageUrl}
                download={`${gen.id}.png`}
                onClick={(e) => e.stopPropagation()}
                className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                title="Download PNG"
              >
                <Download className="w-4 h-4" />
              </a>
              {onDelete && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(gen.id);
                  }}
                  className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-rose-600/90 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                  title="Delete artwork"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Bottom Prompt Excerpt & Specs */}
            <div className="text-white space-y-1">
              {hit && (
                <div className="flex items-center gap-1 text-[10px] text-white/80">
                  {hit.match !== 'text' && <Eye className="w-3 h-3" />}
                  {hit.match !== 'semantic' && <WordsIcon hit={hit} />}
                  <span>{describeHit(hit)}</span>
                </div>
              )}
              <p className="text-xs font-medium line-clamp-2 leading-snug">
                {gen.prompt}
              </p>
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-white/70 pt-0.5">
                <span>seed {gen.params.seed}</span>
                <span>·</span>
                <span>{gen.params.steps} steps</span>
                {generationSeconds(gen.timings) != null && (
                  <>
                    <span>·</span>
                    <span>{formatDuration(generationSeconds(gen.timings)!)}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Gallery Filter & Search Header */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 p-4 rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2a2a26] shadow-xs">
        <div className="flex flex-wrap items-center gap-2">
          {/* Favorites Filter Pill */}
          {onToggleFavorite && (
            <button
              onClick={() => onFiltersChange({ ...filters, favorites: !onlyFavorites })}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                onlyFavorites
                  ? 'bg-rose-500 text-white shadow-xs'
                  : 'bg-[#f1eee7] dark:bg-[#262622] text-[#58423a] dark:text-[#b0a098] hover:bg-[#e5e2db] dark:hover:bg-[#32322c]'
              }`}
            >
              <Heart className={`w-3.5 h-3.5 ${onlyFavorites ? 'fill-current' : ''}`} />
              <span>Favorites</span>
              <span className="font-mono text-[11px] opacity-80">({favoriteCount})</span>
            </button>
          )}

          {/* What the demo shows: the owner's review of what is public */}
          {onToggleDemo && (
            <button
              onClick={() => onFiltersChange({ ...filters, demo: !filters.demo })}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                filters.demo
                  ? 'bg-sky-600 text-white shadow-xs'
                  : 'bg-[#f1eee7] dark:bg-[#262622] text-[#58423a] dark:text-[#b0a098] hover:bg-[#e5e2db] dark:hover:bg-[#32322c]'
              }`}
              aria-pressed={filters.demo}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>In demo</span>
            </button>
          )}

          {/* Aspect Ratio Filter Pills */}
          <div className="flex items-center gap-1 overflow-x-auto py-0.5">
            <button
              onClick={() => onFiltersChange({ ...filters, aspect: '' })}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                selectedRatio === 'all'
                  ? 'bg-[#1c1c18] text-white dark:bg-[#f3f0e9] dark:text-[#1c1c18]'
                  : 'bg-[#f1eee7] dark:bg-[#262622] text-[#58423a] dark:text-[#b0a098] hover:bg-[#e5e2db] dark:hover:bg-[#32322c]'
              }`}
            >
              All ({totalCount})
            </button>
            {(Object.keys(ASPECT_RATIOS) as AspectRatio[]).map((ratio) => {
              const count = aspectCounts.get(ratio) || 0;
              if (count === 0 && selectedRatio !== ratio) return null;
              return (
                <button
                  key={ratio}
                  onClick={() => onFiltersChange({ ...filters, aspect: ratio })}
                  className={`px-2.5 py-1.5 rounded-full text-xs font-mono transition-all ${
                    selectedRatio === ratio
                      ? 'bg-[#1c1c18] text-white dark:bg-[#f3f0e9] dark:text-[#1c1c18]'
                      : 'bg-[#f1eee7] dark:bg-[#262622] text-[#58423a] dark:text-[#b0a098] hover:bg-[#e5e2db] dark:hover:bg-[#32322c]'
                  }`}
                >
                  {ratio} {count > 0 && <span className="opacity-70">({count})</span>}
                </button>
              );
            })}
          </div>
        </div>

        {/* Search Input */}
        <div className="relative min-w-[220px] max-w-sm">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#8c7169] dark:text-[#a09088]" />
          <input
            type="text"
            ref={searchBox}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search images, prompts, seed..."
            className="w-full pl-9 pr-8 py-1.5 rounded-xl text-xs bg-[#f1eee7] dark:bg-[#232320] text-[#1c1c18] dark:text-[#f3f0e9] border border-transparent focus:border-[#a53705] focus:outline-none placeholder-[#8c7169]/70 dark:placeholder-[#a09088]/60 transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => {
                setSearchQuery('');
                searchBox.current?.focus();
              }}
              aria-label="Clear search"
              title="Clear search"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#8c7169] hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* "More like this": the image the rest are ranked against */}
      {filters.like && (
        <div className="flex items-center gap-3 px-3 py-2 rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2a2a26] shadow-xs w-fit max-w-full">
          <a
            href={imageHref(filters.like)}
            onClick={(e) => onLinkClick(e, imageHref(filters.like), { from: 'gallery' })}
            className="shrink-0"
            title="Open this image"
          >
            <img
              src={`/api/images/${encodeURIComponent(filters.like)}/thumb?size=512`}
              alt=""
              className="w-10 h-10 rounded-lg object-cover"
            />
          </a>
          <div className="flex items-center gap-1.5 text-xs text-[#58423a] dark:text-[#b0a098]">
            <ScanSearch className="w-3.5 h-3.5" />
            <span className="font-medium">More like this</span>
          </div>
          <button
            onClick={() => onFiltersChange({ ...filters, like: '' })}
            className="p-1 rounded-lg text-[#8c7169] hover:text-[#1c1c18] dark:hover:text-[#f3f0e9] hover:bg-[#f1eee7] dark:hover:bg-[#262622]"
            title="Show all images"
            aria-label="Clear more like this"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Zero Filtered Results */}
      {!loading && generations.length === 0 && hasActiveFilters && (
        <div className="p-12 text-center rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2a2a26] space-y-4">
          <div className="w-10 h-10 mx-auto rounded-full bg-[#f1eee7] dark:bg-[#262622] flex items-center justify-center text-[#8c7169]">
            <SlidersHorizontal className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-['Plus_Jakarta_Sans'] font-semibold text-base text-[#1c1c18] dark:text-[#f3f0e9]">
              No images match these filters
            </h3>
            <p className="text-xs text-[#8c7169] dark:text-[#a09088] mt-1">
              Try relaxing your aspect ratio or search query to find previous creations.
            </p>
          </div>
          <button
            onClick={resetFilters}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90 transition-opacity"
          >
            Reset filters
          </button>
        </div>
      )}

      {waiting && <PageLoader />}

      {/* Masonry Grid: the previous results, dimmed, until a new search's arrive */}
      {strongerResults.length > 0 && (
        <div className={`columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4 transition-opacity ${loading ? 'opacity-50' : ''}`}>
          {strongerResults.map(renderTile)}
        </div>
      )}

      {/* Found only by a faint resemblance: after a divider */}
      {weakerResults.length > 0 && (
        <>
          <div className="flex items-center gap-3 pt-2" role="separator">
            <div className="h-px flex-1 bg-[#e5e2db] dark:bg-[#2a2a26]" />
            <span className="text-xs text-[#8c7169] dark:text-[#a09088]">Weaker matches</span>
            <div className="h-px flex-1 bg-[#e5e2db] dark:bg-[#2a2a26]" />
          </div>
          <div className={`columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4 transition-opacity ${loading ? 'opacity-50' : ''}`}>
            {weakerResults.map(renderTile)}
          </div>
        </>
      )}
      <LoadMore hasMore={hasMore} loading={loadingMore} onLoadMore={onLoadMore} />
    </div>
  );
};
