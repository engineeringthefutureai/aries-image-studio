import React from 'react';

const ACCENT = '#a53705';
const CREAM = '#fcf9f2';
const GOLD = '#ffc98a';

// Pixel trail behind the ram: [x, y, size, opacity]
const TRAIL: [number, number, number, number][] = [
  [24.9, 25.4, 3.2, 1], [20.45, 25.75, 2.5, 0.87], [16.63, 26.03, 1.95, 0.77], [13.3, 26.24, 1.52, 0.68],
  [23.4, 31.4, 3.2, 1], [18.95, 31.75, 2.5, 0.87], [15.13, 32.03, 1.95, 0.77],
  [21.9, 37.4, 3.2, 1], [17.45, 37.75, 2.5, 0.87], [13.63, 38.03, 1.95, 0.77],
];

/** The app mark: a ram charging out of a pixel trail, inside a picture frame.
 *  Same drawing as public/favicon.svg. */
export const Logo: React.FC<{ className?: string; title?: string }> = ({ className, title }) => (
  <svg
    viewBox="0 0 64 64"
    className={className}
    role={title ? 'img' : undefined}
    aria-label={title}
    aria-hidden={title ? undefined : true}
  >
    <rect width="64" height="64" rx="16" fill={ACCENT} />
    <path
      fill="none"
      stroke={CREAM}
      strokeWidth={4.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M34 14H18a6 6 0 0 0-6 6v24a6 6 0 0 0 6 6h26a6 6 0 0 0 6-6V30"
    />
    <g fill={CREAM}>
      {TRAIL.map(([x, y, s, o]) => (
        <rect key={`${x}-${y}`} x={x} y={y} width={s} height={s} rx={0.5} opacity={o} />
      ))}
      <path d="M33 26C40 25.5 46 31 47.5 38C48.3 41.5 46 44.5 42.5 43.5L36.5 41.5C32.5 40.5 29.5 37.5 29.5 33.5C29.5 29.5 31 26.2 33 26Z" />
      <path d="M57 32.8Q57.36 35.64 60.2 36Q57.36 36.36 57 39.2Q56.64 36.36 53.8 36Q56.64 35.64 57 32.8Z" />
    </g>
    <path
      fill="none"
      stroke={GOLD}
      strokeWidth={3.6}
      strokeLinecap="round"
      d="M37 27C35 22.5 28 22 25.5 26.5C23 31 25.5 37 30.5 37.5C34 37.8 35.5 34.5 34 32.2C32.8 30.5 30.2 31 30 33"
    />
    <circle cx={40} cy={33.5} r={1.3} fill={ACCENT} />
    <path fill={GOLD} d="M45 8Q46.23 17.77 56 19Q46.23 20.23 45 30Q43.77 20.23 34 19Q43.77 17.77 45 8Z" />
  </svg>
);
