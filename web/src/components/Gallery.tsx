import React, { useState, useMemo } from 'react';
import { type Generation, type AspectRatio, ASPECT_RATIOS } from '../types';
import { formatDuration, generationSeconds } from '../utils/timings';
import {
  Heart,
  Dices,
  Maximize2,
  Download,
  Search,
  SlidersHorizontal,
  X,
  Image as ImageIcon,
  Trash2,
} from 'lucide-react';
import { PageLoader } from './PageLoader';

interface GalleryProps {
  generations: Generation[];
  onOpenLightbox: (generation: Generation) => void;
  onToggleFavorite: (id: string) => void;
  onVary: (generation: Generation) => void;
  onDelete?: (id: string) => void;
  /** Studio's first job list has not arrived yet: show a loader, not the empty state. */
  loading?: boolean;
}

/** Column width of the masonry grid (max-w-7xl, px-4/sm:px-6, 1–4 columns, gap-4), for `sizes`. */
const GALLERY_TILE_SIZES =
  '(min-width: 1280px) 296px, (min-width: 1024px) calc((100vw - 80px) / 3), (min-width: 640px) calc((100vw - 64px) / 2), calc(100vw - 32px)';

export const Gallery: React.FC<GalleryProps> = ({
  generations,
  onOpenLightbox,
  onToggleFavorite,
  onVary,
  onDelete,
  loading = false,
}) => {
  const [onlyFavorites, setOnlyFavorites] = useState(false);
  const [selectedRatio, setSelectedRatio] = useState<AspectRatio | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Only consider completed generations with an image
  const completedGenerations = useMemo(() => {
    return generations.filter((g) => g.status === 'done' && g.imageUrl);
  }, [generations]);

  // Counts
  const totalCount = completedGenerations.length;
  const favoriteCount = completedGenerations.filter((g) => g.favorite).length;

  // Filtered generations
  const filtered = useMemo(() => {
    return completedGenerations.filter((g) => {
      if (onlyFavorites && !g.favorite) return false;
      if (selectedRatio !== 'all' && g.params.aspect !== selectedRatio) return false;
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const matchesPrompt = g.prompt.toLowerCase().includes(query);
        const matchesEnhanced = g.enhancedPrompt?.toLowerCase().includes(query);
        const matchesSeed = g.params.seed.toString().includes(query);
        if (!matchesPrompt && !matchesEnhanced && !matchesSeed) return false;
      }
      return true;
    });
  }, [completedGenerations, onlyFavorites, selectedRatio, searchQuery]);

  const hasActiveFilters = onlyFavorites || selectedRatio !== 'all' || searchQuery.trim() !== '';

  const resetFilters = () => {
    setOnlyFavorites(false);
    setSelectedRatio('all');
    setSearchQuery('');
  };

  if (loading && completedGenerations.length === 0) return <PageLoader />;

  if (completedGenerations.length === 0) {
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

  return (
    <div className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
      {/* Gallery Filter & Search Header */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 p-4 rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] border border-[#e5e2db] dark:border-[#2a2a26] shadow-xs">
        <div className="flex flex-wrap items-center gap-2">
          {/* Favorites Filter Pill */}
          <button
            onClick={() => setOnlyFavorites((prev) => !prev)}
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

          {/* Aspect Ratio Filter Pills */}
          <div className="flex items-center gap-1 overflow-x-auto py-0.5">
            <button
              onClick={() => setSelectedRatio('all')}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                selectedRatio === 'all'
                  ? 'bg-[#1c1c18] text-white dark:bg-[#f3f0e9] dark:text-[#1c1c18]'
                  : 'bg-[#f1eee7] dark:bg-[#262622] text-[#58423a] dark:text-[#b0a098] hover:bg-[#e5e2db] dark:hover:bg-[#32322c]'
              }`}
            >
              All ({totalCount})
            </button>
            {(Object.keys(ASPECT_RATIOS) as AspectRatio[]).map((ratio) => {
              const count = completedGenerations.filter((g) => g.params.aspect === ratio).length;
              if (count === 0 && selectedRatio !== ratio) return null;
              return (
                <button
                  key={ratio}
                  onClick={() => setSelectedRatio(ratio)}
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
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search prompts or seed..."
            className="w-full pl-9 pr-8 py-1.5 rounded-xl text-xs bg-[#f1eee7] dark:bg-[#232320] text-[#1c1c18] dark:text-[#f3f0e9] border border-transparent focus:border-[#a53705] focus:outline-none placeholder-[#8c7169]/70 dark:placeholder-[#a09088]/60 transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#8c7169] hover:text-[#1c1c18] dark:hover:text-[#f3f0e9]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Zero Filtered Results */}
      {filtered.length === 0 && hasActiveFilters && (
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

      {/* Masonry Grid */}
      <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
        {filtered.map((gen) => {
          const aspectConfig = ASPECT_RATIOS[gen.params.aspect] || ASPECT_RATIOS['1:1'];
          const aspectRatioStyle = `${aspectConfig.ratioWidth} / ${aspectConfig.ratioHeight}`;

          return (
            <div
              key={gen.id}
              onClick={() => onOpenLightbox(gen)}
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

                {/* Top Badges */}
                <div className="absolute top-2.5 right-2.5 z-20 flex items-center gap-1 bg-black/50 backdrop-blur-md px-2 py-0.5 rounded-full text-[10px] font-mono text-white/90">
                  <span>{gen.params.aspect}</span>
                </div>

                {gen.favorite && (
                  <div className="absolute top-2.5 left-2.5 z-20 w-6 h-6 rounded-full bg-rose-500 text-white flex items-center justify-center shadow-xs">
                    <Heart className="w-3.5 h-3.5 fill-current" />
                  </div>
                )}

                {/* Hover Scrim & Quick Action HUD */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-3.5 z-10">
                  {/* Top Buttons */}
                  <div className="flex items-center justify-end gap-1.5">
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
                        onOpenLightbox(gen);
                      }}
                      className="w-8 h-8 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md flex items-center justify-center transition-transform hover:scale-110"
                      title="Inspect full artwork"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
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
        })}
      </div>
    </div>
  );
};
