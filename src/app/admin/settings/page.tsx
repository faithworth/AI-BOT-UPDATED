'use client';
// ============================================================
// VUKA — Admin Platform Settings (Phase 5)
// /admin/settings — subscription plans, payout settings,
// DSP platforms, genre tags, landing page content, feature flags.
// ============================================================

import { useEffect, useState, useCallback } from 'react';
import { Loader2, RefreshCw, Save, Plus, Trash2, Check, X, ToggleLeft, ToggleRight } from 'lucide-react';

type SettingsTab = 'plans' | 'payouts' | 'platforms' | 'genres' | 'landing' | 'flags';

export default function AdminSettingsPage() {
  const [tab, setTab]         = useState<SettingsTab>('plans');
  const [settings, setSettings] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving]   = useState(false);
  const [saved, setSaved]     = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/admin/settings');
      if (res.ok) setSettings(await res.json());
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function save(section: string, data: any) {
    setSaving(true);
    try {
      const res = await fetch('/api/admin/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section, data }),
      });
      if (res.ok) { setSaved(true); setTimeout(() => setSaved(false), 2000); await load(); }
      else { const d = await res.json(); alert(d.error || 'Save failed'); }
    } finally { setSaving(false); }
  }

  const TABS: { key: SettingsTab; label: string }[] = [
    { key: 'plans',    label: 'Subscription Plans' },
    { key: 'payouts',  label: 'Payout Settings' },
    { key: 'platforms', label: 'DSP Platforms' },
    { key: 'genres',   label: 'Genre Tags' },
    { key: 'landing',  label: 'Landing Page' },
    { key: 'flags',    label: 'Feature Flags' },
  ];

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black font-display">Platform Settings</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Edit all platform configuration — no code deploys needed</p>
        </div>
        <div className="flex items-center gap-3">
          {saved && (
            <span className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--green)' }}>
              <Check size={14} /> Saved
            </span>
          )}
          <button onClick={load} disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}>
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>
      </div>

      <div className="flex gap-2 flex-wrap mb-6">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className="px-4 py-2 rounded-xl text-sm font-medium"
            style={{
              background: tab === t.key ? 'rgba(160,232,124,0.12)' : 'var(--surface)',
              color: tab === t.key ? 'var(--green)' : 'var(--text-muted)',
              border: tab === t.key ? '1px solid rgba(160,232,124,0.3)' : '1px solid var(--border)',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="animate-spin" size={24} style={{ color: 'var(--green)' }} />
        </div>
      ) : (
        <div className="rounded-2xl p-6" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

          {/* Subscription Plans */}
          {tab === 'plans' && (
            <div className="space-y-4">
              <div>
                <h2 className="font-bold text-lg mb-1">Subscription Plans</h2>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  Changes here update the database. New signups and Paystack billing use these plan definitions.
                </p>
              </div>
              {(settings.plans || []).map((plan: any, i: number) => (
                <div key={plan.id || i} className="p-4 rounded-xl" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                    <div>
                      <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Plan Name</label>
                      <input value={plan.name || ''} onChange={e => {
                        const p = [...(settings.plans || [])]; p[i] = { ...p[i], name: e.target.value };
                        setSettings((s: any) => ({ ...s, plans: p }));
                      }} className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                        style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                    </div>
                    <div>
                      <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Price (ZAR)</label>
                      <input type="number" value={plan.priceZAR || 0} onChange={e => {
                        const p = [...(settings.plans || [])]; p[i] = { ...p[i], priceZAR: parseFloat(e.target.value) };
                        setSettings((s: any) => ({ ...s, plans: p }));
                      }} className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                        style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                    </div>
                    <div>
                      <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Price (USD)</label>
                      <input type="number" value={plan.priceUSD || 0} onChange={e => {
                        const p = [...(settings.plans || [])]; p[i] = { ...p[i], priceUSD: parseFloat(e.target.value) };
                        setSettings((s: any) => ({ ...s, plans: p }));
                      }} className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                        style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                    </div>
                    <div>
                      <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Royalty % (artist keeps)</label>
                      <input type="number" min={0} max={100} value={plan.royaltyShare || 100} onChange={e => {
                        const p = [...(settings.plans || [])]; p[i] = { ...p[i], royaltyShare: parseFloat(e.target.value) };
                        setSettings((s: any) => ({ ...s, plans: p }));
                      }} className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                        style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span>Slug: <code>{plan.slug}</code></span>
                    <span>Billing: {plan.billingPeriod}</span>
                  </div>
                </div>
              ))}
              <button onClick={() => save('plans', settings.plans)}
                disabled={saving}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
                style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                Save Plans
              </button>
            </div>
          )}

          {/* Payout Settings */}
          {tab === 'payouts' && (
            <div className="space-y-5 max-w-md">
              <h2 className="font-bold text-lg">Payout Settings</h2>
              {[
                { key: 'minPayoutAmount', label: 'Minimum Payout Amount (ZAR)', type: 'number' },
                { key: 'payoutProcessingDays', label: 'Processing Days', type: 'number' },
                { key: 'paystackPayoutEmail', label: 'Paystack Payout Email', type: 'email' },
              ].map(({ key, label, type }) => (
                <div key={key}>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>{label}</label>
                  <input type={type}
                    value={settings.payouts?.[key] || ''}
                    onChange={e => setSettings((s: any) => ({
                      ...s, payouts: { ...(s.payouts || {}), [key]: type === 'number' ? parseFloat(e.target.value) : e.target.value }
                    }))}
                    className="w-full px-3 py-2.5 rounded-xl text-sm outline-none"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                </div>
              ))}
              <button onClick={() => save('payouts', settings.payouts)}
                disabled={saving}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
                style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                Save Payout Settings
              </button>
            </div>
          )}

          {/* DSP Platforms */}
          {tab === 'platforms' && (
            <div className="space-y-4">
              <h2 className="font-bold text-lg mb-4">Distribution Platforms</h2>
              <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--border)' }}>
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ background: 'var(--bg)', borderBottom: '1px solid var(--border)' }}>
                      {['Platform', 'Slug', 'Delivery Days', 'Active', 'Actions'].map(h => (
                        <th key={h} className="px-4 py-3 text-left font-semibold" style={{ color: 'var(--text-muted)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(settings.platforms || []).map((p: any, i: number) => (
                      <tr key={p.id || i} className="border-t" style={{ borderColor: 'var(--border)' }}>
                        <td className="px-4 py-2.5 font-medium">{p.name}</td>
                        <td className="px-4 py-2.5 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{p.slug}</td>
                        <td className="px-4 py-2.5">{p.avgDeliveryDays}d</td>
                        <td className="px-4 py-2.5">
                          <span className="px-2 py-0.5 rounded-full text-xs font-semibold"
                            style={{ background: p.isActive ? 'rgba(160,232,124,0.15)' : 'rgba(255,77,77,0.15)',
                              color: p.isActive ? 'var(--green)' : '#ff4d4d' }}>
                            {p.isActive ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="px-4 py-2.5">
                          <button onClick={() => save('toggle_platform', { id: p.id, isActive: !p.isActive })}
                            className="text-xs px-3 py-1 rounded-lg"
                            style={{ background: 'var(--surface)', color: 'var(--text-muted)' }}>
                            Toggle
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Feature Flags */}
          {tab === 'flags' && (
            <div className="space-y-4 max-w-lg">
              <h2 className="font-bold text-lg">Feature Flags</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Toggle platform features on or off without code deploys.</p>
              {[
                { key: 'enableRegistration',    label: 'New Registrations', desc: 'Allow new artists to register' },
                { key: 'enableDistribution',    label: 'Distribution',      desc: 'Enable DSP distribution engine' },
                { key: 'enableBeatStore',       label: 'Beat Store',        desc: 'Beat marketplace for producers' },
                { key: 'enableVideoUpload',     label: 'Video Uploads',     desc: 'Video content distribution' },
                { key: 'enableMemberships',     label: 'Memberships',       desc: 'Fan membership / creator tiers' },
                { key: 'enableMaintenanceMode', label: 'Maintenance Mode',  desc: 'Show maintenance page to all users' },
              ].map(f => {
                const val = settings.flags?.[f.key] ?? true;
                return (
                  <div key={f.key} className="flex items-center justify-between p-4 rounded-xl"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                    <div>
                      <div className="font-medium text-sm">{f.label}</div>
                      <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{f.desc}</div>
                    </div>
                    <button onClick={() => {
                      const updated = { ...(settings.flags || {}), [f.key]: !val };
                      setSettings((s: any) => ({ ...s, flags: updated }));
                      save('flags', updated);
                    }}
                      style={{ color: val ? 'var(--green)' : 'var(--text-muted)' }}>
                      {val ? <ToggleRight size={28} /> : <ToggleLeft size={28} />}
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {/* Landing Page */}
          {tab === 'landing' && (
            <div className="space-y-5 max-w-2xl">
              <h2 className="font-bold text-lg">Landing Page Content</h2>
              {[
                { key: 'heroHeadline', label: 'Hero Headline', type: 'text' },
                { key: 'heroSubtext', label: 'Hero Subtext', type: 'textarea' },
              ].map(({ key, label, type }) => (
                <div key={key}>
                  <label className="block text-sm font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>{label}</label>
                  {type === 'textarea'
                    ? <textarea value={settings.landing?.[key] || ''} rows={3}
                        onChange={e => setSettings((s: any) => ({ ...s, landing: { ...(s.landing || {}), [key]: e.target.value } }))}
                        className="w-full px-3 py-2.5 rounded-xl text-sm resize-none outline-none"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                    : <input type="text" value={settings.landing?.[key] || ''}
                        onChange={e => setSettings((s: any) => ({ ...s, landing: { ...(s.landing || {}), [key]: e.target.value } }))}
                        className="w-full px-3 py-2.5 rounded-xl text-sm outline-none"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                  }
                </div>
              ))}
              <button onClick={() => save('landing', settings.landing)}
                disabled={saving}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
                style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
                Save Landing Page
              </button>
            </div>
          )}

          {/* Genres */}
          {tab === 'genres' && (
            <div className="space-y-4 max-w-lg">
              <h2 className="font-bold text-lg">Genre Tags</h2>
              <div className="flex flex-wrap gap-2">
                {(settings.genres || [
                  'Amapiano', 'Afrobeats', 'Gqom', 'Hip-Hop', 'R&B',
                  'Drill', 'House', 'Kwaito', 'Gospel', 'Jazz', 'Pop', 'Electronic',
                ]).map((g: string) => (
                  <span key={g} className="flex items-center gap-1 px-3 py-1.5 rounded-xl text-sm"
                    style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)', border: '1px solid rgba(160,232,124,0.2)' }}>
                    {g}
                    <button onClick={() => setSettings((s: any) => ({
                      ...s, genres: (s.genres || []).filter((x: string) => x !== g)
                    }))} className="ml-1 opacity-60 hover:opacity-100">
                      <X size={10} />
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input id="newGenre" placeholder="Add new genre…"
                  className="flex-1 px-3 py-2.5 rounded-xl text-sm outline-none"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      const val = (e.target as HTMLInputElement).value.trim();
                      if (val) { setSettings((s: any) => ({ ...s, genres: [...(s.genres || []), val] })); (e.target as HTMLInputElement).value = ''; }
                    }
                  }} />
                <button onClick={() => save('genres', settings.genres)}
                  disabled={saving}
                  className="px-4 py-2.5 rounded-xl text-sm font-bold"
                  style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                  <Save size={14} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
