import React, { useEffect, useState } from 'react';
import { Check, Copy, KeyRound, Plug, Trash2, X } from 'lucide-react';
import { createTokenApi, listTokensApi, revokeTokenApi, type ApiToken } from '../services/api';

interface ConnectionsPageProps {
  onClose: () => void;
  /** Studio has sign-in: tokens exist only then (without it, nothing needs one). */
  signIn: boolean;
}

const when = (s: number | null) =>
  s ? new Date(s * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : 'never';

/** Copies `text`, and says so for a moment. */
const CopyButton: React.FC<{ text: string; label: string }> = ({ text, label }) => {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
      className="shrink-0 p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 text-[#58423a] dark:text-[#a09088]"
      title={label}
      aria-label={label}
    >
      {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
    </button>
  );
};

const Code: React.FC<{ text: string; label: string }> = ({ text, label }) => (
  <div className="flex items-start gap-2 rounded-lg bg-[#f1eee7] dark:bg-[#20201d] px-3 py-2">
    <pre className="flex-1 min-w-0 overflow-x-auto text-xs font-mono whitespace-pre">{text}</pre>
    <CopyButton text={text} label={label} />
  </div>
);

/**
 * `/connections`: Studio's MCP address, and the access tokens that let the CLI and AI tools
 * act as this account (docs/INTEGRATIONS.md). A new token is shown once.
 */
export const ConnectionsPage: React.FC<ConnectionsPageProps> = ({ onClose, signIn }) => {
  const [tokens, setTokens] = useState<ApiToken[] | null>(null);
  const [name, setName] = useState('');
  const [made, setMade] = useState<{ token: string; name: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const mcpUrl = `${window.location.origin}/mcp`;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    if (signIn) listTokensApi().then(setTokens);
  }, [signIn]);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setError(null);
    const res = await createTokenApi(name.trim());
    setBusy(false);
    if ('error' in res) {
      setError(res.error);
      return;
    }
    setMade({ token: res.token, name: res.record.name });
    setName('');
    setTokens((t) => [res.record, ...(t || [])]);
  };

  const revoke = async (t: ApiToken) => {
    if (!window.confirm(`Revoke "${t.name}"? Whatever uses it stops working at once.`)) return;
    if (await revokeTokenApi(t.id)) setTokens((all) => (all || []).filter((x) => x.id !== t.id));
  };

  const token = made?.token ?? '<token>';
  const antigravity = JSON.stringify(
    { mcpServers: { 'aries-studio': { serverUrl: mcpUrl, headers: { Authorization: `Bearer ${token}` } } } },
    null,
    2,
  );

  const section = 'text-xs font-bold uppercase tracking-wider text-[#8c7169] dark:text-[#a09088] mb-3';
  return (
    <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-6">
      <div className="w-full rounded-2xl bg-[#fcf9f2] dark:bg-[#1a1a17] text-[#1c1c18] dark:text-[#f3f0e9] border border-[#e5e2db] dark:border-[#2e2e2a] shadow-xs overflow-hidden">
        <div className="px-6 py-4 border-b border-[#e5e2db] dark:border-[#2e2e2a] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Plug className="w-5 h-5 text-[#a53705]" />
            <h3 className="font-['Plus_Jakarta_Sans'] font-bold text-base">Connections · CLI &amp; AI tools</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 text-[#58423a] dark:text-[#a09088]"
            title="Close (Esc)"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-8 text-sm">
          <section>
            <h4 className={section}>MCP server</h4>
            <p className="mb-2 text-[#58423a] dark:text-[#a09088]">
              AI tools that speak MCP make, find and show your images here.
            </p>
            <Code text={mcpUrl} label="Copy the MCP address" />
          </section>

          {signIn ? (
            <section>
              <h4 className={section}>Access tokens</h4>
              <p className="mb-3 text-[#58423a] dark:text-[#a09088]">
                A token lets the CLI or an AI tool act as you: it sees and makes what you do. Give each tool its
                own, and revoke one you no longer use.
              </p>
              <form onSubmit={create} className="flex gap-2 mb-4">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Name it: “Antigravity on my laptop”"
                  maxLength={80}
                  aria-label="Token name"
                  className="flex-1 min-w-0 rounded-lg border border-[#e5e2db] dark:border-[#2d2d2a] bg-white dark:bg-[#121210] px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-[#a53705]/40"
                />
                <button
                  type="submit"
                  disabled={!name.trim() || busy}
                  className="px-4 py-1.5 rounded-lg text-sm font-medium bg-[#a53705] text-white hover:opacity-90 disabled:opacity-40"
                >
                  Create
                </button>
              </form>
              {error && <p className="mb-3 text-rose-700 dark:text-rose-300">{error}</p>}
              {made && (
                <div className="mb-4 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 space-y-2">
                  <p className="font-medium">
                    “{made.name}”: copy it now. Studio keeps no copy and cannot show it again.
                  </p>
                  <Code text={made.token} label="Copy the token" />
                </div>
              )}
              {tokens === null ? (
                <p className="text-[#8c7169]">Loading…</p>
              ) : tokens.length === 0 ? (
                <p className="text-[#8c7169]">No tokens yet.</p>
              ) : (
                <ul className="divide-y divide-[#e5e2db] dark:divide-[#2d2d2a] rounded-xl border border-[#e5e2db] dark:border-[#2d2d2a]">
                  {tokens.map((t) => (
                    <li key={t.id} className="flex items-center gap-3 px-3 py-2">
                      <KeyRound className="w-4 h-4 shrink-0 text-[#8c7169]" />
                      <div className="flex-1 min-w-0">
                        <div className="truncate font-medium">{t.name}</div>
                        <div className="text-xs text-[#8c7169] dark:text-[#a09088]">
                          Created {when(t.created_at)} · last used {when(t.last_used_at)}
                        </div>
                      </div>
                      <button
                        onClick={() => revoke(t)}
                        className="p-1.5 rounded-lg hover:bg-rose-500/10 text-rose-700 dark:text-rose-300"
                        title="Revoke"
                        aria-label={`Revoke ${t.name}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          ) : (
            <p className="text-[#58423a] dark:text-[#a09088]">
              This Studio has no sign-in, so tools need no token: anyone who can reach it can use it.
            </p>
          )}

          <section>
            <h4 className={section}>Set up</h4>
            <div className="space-y-4">
              <div>
                <p className="mb-1.5 font-medium">The aries CLI</p>
                <Code
                  text={`aries login ${window.location.origin}${signIn ? ' --token ' + token : ''}`}
                  label="Copy the command"
                />
              </div>
              <div>
                <p className="mb-1.5 font-medium">Antigravity: add the server</p>
                <Code text={`agy mcp add aries-studio ${mcpUrl}`} label="Copy the command" />
              </div>
              <div>
                <p className="mb-1.5 font-medium">Antigravity: ~/.gemini/config/mcp_config.json</p>
                <Code text={antigravity} label="Copy the configuration" />
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
