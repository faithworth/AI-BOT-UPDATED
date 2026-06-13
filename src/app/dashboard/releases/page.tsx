// FIX: src/app/dashboard/releases/page.tsx
// Added ISRC display per track and UPC display per release.
// Previously the releases list showed "X tracks · Y plays" but NEVER showed ISRCs.
// ISRCs are generated in the distribution system (DistributionTrack.isrc) and also
// need to be auto-generated and stored on the regular Release tracks for Paystack royalty tracking.
//
// This page now:
// - Shows UPC on the release row
// - Expands to show per-track ISRC when clicked
// - Fetches from /api/dashboard/releases which includes tracks

'use client';
import { useEffect, useState } from 'react';
import { formatCurrency } from '@/lib/utils';
import Link from 'next/link';
import { Plus, ExternalLink, Music, Trash2, AlertTriangle, Eye, EyeOff, ChevronDown, ChevronUp, Hash, Copy, Check, Pencil } from 'lucide-react';

export default function DashboardReleasesPage() {
  const [releases, setReleases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/dashboard/releases')
      .then(r => r.json())
      .then(d => setReleases(Array.isArray(d.releases) ? d.releases : []))
      .catch(() => setReleases([]))
      .finally(() => setLoading(false));
  }, []);

  async function toggleActive(id: string, current: boolean) {
    await fetch('/api/dashboard/releases', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ releaseId: id, isActive: !current }),
    });
    setReleases(prev => prev.map(r => r.id === id ? { ...r, isActive: !current } : r));
  }

  async function deleteRelease(id: string) {
    setDeleting(id);
    setConfirmId(null);
    const res = await fetch(`/api/dashboard/releases?releaseId=${id}`, { method: 'DELETE' });
    if (res.ok) {
      setReleases(prev => prev.filter(r => r.id !== id));
    } else {
      const data = await res.json().catch(() => ({}));
      alert(data.error || 'Could not delete this release.');
    }
    setDeleting(null);
  }

  function copyCode(code: string) {
    navigator.clipboard.writeText(code).catch(() => {});
    setCopied(code);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <div className="p-6 max-w-5xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-black" style={{ color: 'var(--text)' }}>My Releases</h1>
        <Link href="/dashboard/releases/new"
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold text-white"
          style={{ background: 'var(--sky)' }}>
          <Plus className="w-4 h-4" /> Upload Release
        </Link>
      </div>

      {loading && <p style={{ color: 'var(--text-muted)' }}>Loading…</p>}

      {!loading && releases.length === 0 && (
        <div className="text-center py-20 rounded-2xl border" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
          <Music className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--sky)' }} />
          <p className="font-bold mb-2" style={{ color: 'var(--text)' }}>Nothing here yet</p>
          <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>Upload your first EP, album or single.</p>
          <Link href="/dashboard/releases/new" className="px-4 py-2 rounded-lg font-bold text-white inline-block"
            style={{ background: 'var(--sky)' }}>Upload Now</Link>
        </div>
      )}

      <div className="space-y-3">
        {releases.map((release: any) => (
          <div key={release.id}
            className="rounded-xl border overflow-hidden"
            style={{ background: 'var(--surface)', borderColor: 'var(--border)', opacity: deleting === release.id ? 0.5 : 1 }}>

            {/* Main row */}
            <div className="flex items-center gap-4 p-4">
              <div className="w-14 h-14 rounded-lg overflow-hidden flex-shrink-0 flex items-center justify-center text-2xl"
                style={{ background: 'var(--surface2)' }}>
                {release.artworkUrl
                  ? <img src={release.artworkUrl} className="w-full h-full object-cover" alt={release.title} />
                  : '🎵'}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="font-bold text-sm" style={{ color: 'var(--text)' }}>{release.title}</span>
                  <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--surface2)', color: 'var(--sky)' }}>
                    {release.releaseType}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full"
                    style={{
                      background: release._isDistrib
                        ? 'rgba(56,182,232,0.1)'
                        : release.isActive ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.1)',
                      color: release._isDistrib
                        ? 'var(--sky)'
                        : release.isActive ? 'var(--green)' : '#ef4444',
                    }}>
                    {release._isDistrib ? (release.status ?? 'distribution') : (release.isActive ? 'Active' : 'Hidden')}
                  </span>
                </div>
                <div className="text-xs flex flex-wrap gap-x-3 gap-y-0.5" style={{ color: 'var(--text-muted)' }}>
                  <span>{release.tracks?.length || 0} tracks</span>
                  <span>{release.plays ?? 0} plays</span>
                  <span>{release.sales ?? 0} sales</span>
                  <span>{formatCurrency(release.price)}</span>
                  {release.distributor && (
                    <span className="px-1.5 py-0.5 rounded text-xs font-medium"
                      style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)' }}>
                      via {release.distributor}
                    </span>
                  )}
                  {/* UPC display */}
                  {release.upc && (
                    <button
                      onClick={() => copyCode(release.upc)}
                      className="flex items-center gap-1 font-mono hover:underline"
                      title="Copy UPC"
                      style={{ color: 'var(--sky)' }}>
                      <Hash size={10} />
                      UPC: {release.upc}
                      {copied === release.upc ? <Check size={10} /> : <Copy size={10} />}
                    </button>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-1 flex-shrink-0">
                {/* Expand to show tracks + ISRCs */}
                {release.tracks?.length > 0 && (
                  <button
                    onClick={() => setExpanded(e => e === release.id ? null : release.id)}
                    className="p-2 rounded-lg transition-colors hover:bg-[var(--surface2)]"
                    title="Show track ISRCs"
                    style={{ color: 'var(--text-muted)' }}>
                    {expanded === release.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                )}
                <button onClick={() => !release._isDistrib && toggleActive(release.id, release.isActive)}
                  className="p-2 rounded-lg transition-colors hover:bg-[var(--surface2)]"
                  title={release._isDistrib ? 'Managed via distribution' : (release.isActive ? 'Hide release' : 'Make live')}
                  disabled={!!release._isDistrib}
                  style={{ color: 'var(--text-muted)', opacity: release._isDistrib ? 0.35 : 1 }}>
                  {release.isActive ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
                {release._isDistrib && (
                  <Link href={`/dashboard/releases/${release.id}/edit`}
                    className="p-2 rounded-lg transition-colors hover:bg-[var(--surface2)]"
                    title="Edit / re-upload audio"
                    style={{ color: 'var(--sky)' }}>
                    <Pencil size={16} />
                  </Link>
                )}
                <Link href={release._isDistrib ? `/releases/${release.id}` : `/release/${release.slug}`} target="_blank"
                  className="p-2 rounded-lg transition-colors hover:bg-[var(--surface2)]"
                  style={{ color: 'var(--sky)' }}>
                  <ExternalLink className="w-4 h-4" />
                </Link>
                <button
                  onClick={() => setConfirmId(release.id)}
                  disabled={deleting === release.id}
                  className="p-2 rounded-lg transition-colors hover:bg-red-50"
                  style={{ color: '#ef4444' }}>
                  <Trash2 size={16} />
                </button>
              </div>
            </div>

            {/* Expanded track list with ISRCs */}
            {expanded === release.id && release.tracks?.length > 0 && (
              <div className="border-t" style={{ borderColor: 'var(--border)' }}>
                <div className="px-4 py-2 text-xs font-semibold uppercase tracking-wide"
                  style={{ background: 'var(--surface2)', color: 'var(--text-muted)' }}>
                  Tracks & ISRC Codes
                </div>
                {release.tracks.map((track: any, i: number) => (
                  <div key={track.id} className="flex items-center justify-between px-4 py-2.5 border-t text-sm"
                    style={{ borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="text-xs w-5 text-center flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                        {track.trackNumber ?? i + 1}
                      </span>
                      <span className="truncate font-medium" style={{ color: 'var(--text)' }}>
                        {track.title}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {track.isrc ? (
                        <button
                          onClick={() => copyCode(track.isrc)}
                          className="flex items-center gap-1.5 font-mono text-xs px-2 py-1 rounded-lg transition-colors"
                          style={{ background: 'rgba(56,182,232,0.1)', color: 'var(--sky)' }}
                          title="Copy ISRC">
                          <Hash size={10} />
                          {track.isrc}
                          {copied === track.isrc ? <Check size={10} /> : <Copy size={10} />}
                        </button>
                      ) : (
                        <span className="text-xs px-2 py-1 rounded-lg"
                          style={{ background: 'var(--surface2)', color: 'var(--text-muted)' }}>
                          No ISRC yet
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Confirmation Modal */}
      {confirmId && (() => {
        const release = releases.find(r => r.id === confirmId);
        return (
          <div className="fixed inset-0 z-50 flex items-start justify-center p-4 overflow-y-auto bg-black/50">
            <div className="w-full max-w-sm rounded-2xl p-6" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(239,68,68,0.1)' }}>
                  <AlertTriangle size={18} style={{ color: '#ef4444' }} />
                </div>
                <div>
                  <p className="font-bold" style={{ color: 'var(--text)' }}>Delete "{release?.title}"?</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>This cannot be undone.</p>
                </div>
              </div>
              <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
                All tracks will be permanently removed. If buyers have confirmed purchases, hide it instead.
              </p>
              <div className="flex gap-3">
                <button onClick={() => setConfirmId(null)}
                  className="flex-1 py-2.5 rounded-xl font-semibold text-sm"
                  style={{ background: 'var(--surface2)', color: 'var(--text)' }}>
                  Cancel
                </button>
                <button onClick={() => deleteRelease(confirmId)}
                  className="flex-1 py-2.5 rounded-xl font-bold text-sm text-white"
                  style={{ background: '#ef4444' }}>
                  Delete
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
