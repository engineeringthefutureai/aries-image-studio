import React, { useEffect, useRef, useState } from 'react';
import { TriangleAlert } from 'lucide-react';
import { useBodyScrollLock } from '../hooks/useBodyScrollLock';
import type { Generation } from '../types';
import { uploadThumbUrl } from '../utils/references';

export interface ShareRequest {
  images: Generation[];
  /** Which of them have uploaded images behind them. */
  warned: string[];
  /** Those uploaded images. */
  uploaded: string[];
}

interface DemoShareDialogProps {
  /** null: closed. */
  request: ShareRequest | null;
  onConfirm: () => void;
  onCancel: () => void;
}

const RISKS: [string, string][] = [
  ['Anyone can see it.', 'No sign-in is needed. The image, its prompts and the reference above are public, and can be copied, saved and indexed.'],
  ["Taking it out doesn't take it back.", 'Studio stops showing it at once, but copies that were already made stay out there.'],
  ['A likeness carries over.', 'Faces, children, homes, license plates and documents in a photo can be recognized in the reference and in the result.'],
  ['The people in it did not agree to this.', 'Publishing someone without consent can harm them and can break privacy or image-rights law, wherever they live.'],
  ['The prompt is a recipe.', 'Next to the photo, it shows others how to alter that person or place. Edits of real people can be used to impersonate or harass them.'],
  ['The photo may not be yours to publish.', "Other people's photos are usually under copyright."],
];

/** The images shown as references of these, which become public with them. */
const referencesOf = (images: Generation[]) => [
  ...new Set(images.flatMap((g) => (g.references ?? []).map((r) => r.sha256))),
];

/**
 * Asked every time something is about to be shown in the demo (docs/DEMO.md §3). A plain yes/no,
 * unless uploaded images are behind it: then a warning, where going on takes ticking what the
 * owner is vouching for. Either way the safe answer has the focus.
 */
export const DemoShareDialog: React.FC<DemoShareDialogProps> = ({ request, onConfirm, onCancel }) => {
  if (!request) return null;
  const Dialog = request.uploaded.length ? Warning : Confirmation;
  return <Dialog request={request} onConfirm={onConfirm} onCancel={onCancel} />;
};

interface OpenDialogProps extends DemoShareDialogProps {
  request: ShareRequest;
}

/** Esc cancels, and the page's shortcuts never see a key meant for the dialog (capture phase). */
function useDialogFocus(onCancel: () => void) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  useBodyScrollLock(true);
  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      e.stopPropagation();
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onCancel]);
  return cancelRef;
}

const Confirmation: React.FC<OpenDialogProps> = ({ request, onConfirm, onCancel }) => {
  const cancelRef = useDialogFocus(onCancel);
  const count = request.images.length;
  const references = referencesOf(request.images);
  return (
    <div
      onClick={onCancel}
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="share-confirm-title"
        aria-describedby="share-confirm-message"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-sm max-h-[90dvh] overflow-y-auto rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-2xl p-6"
      >
        <h3 id="share-confirm-title" className="font-['Plus_Jakarta_Sans'] font-bold text-base">
          {count === 1 ? 'Share this image in the demo?' : `Share these ${count} images in the demo?`}
        </h3>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {request.images.map((gen) => (
            <img
              key={gen.id}
              src={gen.thumbUrl || gen.imageUrl}
              alt=""
              className="w-16 h-16 rounded-lg object-cover bg-[#f1eee7] dark:bg-[#232320]"
            />
          ))}
        </div>
        <p id="share-confirm-message" className="mt-3 text-sm text-[#58423a] dark:text-[#a09088]">
          Anyone can then see {count === 1 ? 'it and its prompts' : 'them and their prompts'} without signing in
          {references.length > 0 && ', and these reference images too'}.
        </p>
        {references.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5" role="list" aria-label="Reference images">
            {references.map((sha) => (
              <img
                key={sha}
                role="listitem"
                src={uploadThumbUrl(sha)}
                alt="Reference image"
                className="w-16 h-16 rounded-lg object-cover bg-[#f1eee7] dark:bg-[#232320]"
              />
            ))}
          </div>
        )}
        <div className="mt-6 flex justify-end gap-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#f1eee7] dark:bg-[#2c2c28] hover:bg-[#e5e2db] dark:hover:bg-[#363632] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#a53705] text-white hover:opacity-90 transition-opacity"
          >
            Share
          </button>
        </div>
      </div>
    </div>
  );
};

/** Mounted only while open, so the box starts unticked every time. */
const Warning: React.FC<OpenDialogProps> = ({ request, onConfirm, onCancel }) => {
  const { images, warned, uploaded } = request;
  const shown = [...new Set([...uploaded, ...referencesOf(images)])];
  const [vouched, setVouched] = useState(false);
  const cancelRef = useDialogFocus(onCancel);

  return (
    <div
      onClick={onCancel}
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="share-warning-title"
        aria-describedby="share-warning-summary"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg max-h-[90dvh] overflow-y-auto rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border-2 border-rose-500/60 shadow-2xl p-6"
      >
        <div className="flex items-start gap-3">
          <TriangleAlert className="w-6 h-6 shrink-0 text-rose-600 dark:text-rose-400" />
          <div>
            <h3 id="share-warning-title" className="font-['Plus_Jakarta_Sans'] font-bold text-base">
              {images.length === 1
                ? 'Not recommended: this image was made from an uploaded image'
                : `Not recommended: ${warned.length} of these ${images.length} images ${warned.length === 1 ? 'was' : 'were'} made from an uploaded image`}
            </h3>
            <p id="share-warning-summary" className="mt-1 text-sm text-[#58423a] dark:text-[#a09088]">
              Don't share it if the upload is a photo of a real person or anything private. Sharing it makes the
              upload public too.{images.length > 1 && ` Going on shares all ${images.length} images.`}
            </p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-1.5" role="list" aria-label="Uploaded and reference images">
          {shown.map((sha) => (
            <img
              key={sha}
              role="listitem"
              src={uploadThumbUrl(sha)}
              alt="Uploaded or reference image"
              className="w-16 h-16 rounded-lg object-cover bg-[#f1eee7] dark:bg-[#232320]"
            />
          ))}
        </div>

        <ul className="mt-4 space-y-2 text-sm">
          {RISKS.map(([risk, detail]) => (
            <li key={risk}>
              <span className="font-semibold text-rose-700 dark:text-rose-300">{risk}</span>{' '}
              <span className="text-[#58423a] dark:text-[#a09088]">{detail}</span>
            </li>
          ))}
        </ul>

        <label className="mt-5 flex items-start gap-2.5 p-3 rounded-xl bg-rose-500/10 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={vouched}
            onChange={(e) => setVouched(e.target.checked)}
            className="mt-0.5 w-4 h-4 accent-rose-600"
          />
          <span>
            I checked: the upload was generated or is free to publish, and shows no real person without their
            consent and nothing private.
          </span>
        </label>

        <div className="mt-5 flex justify-end gap-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-[#f1eee7] dark:bg-[#2c2c28] hover:bg-[#e5e2db] dark:hover:bg-[#363632] transition-colors"
          >
            Don't share
          </button>
          <button
            onClick={onConfirm}
            disabled={!vouched}
            className="px-4 py-1.5 rounded-lg text-xs font-medium bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-40 disabled:hover:bg-rose-600 transition-colors"
          >
            Share publicly anyway
          </button>
        </div>
      </div>
    </div>
  );
};
