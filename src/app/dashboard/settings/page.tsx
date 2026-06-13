'use client';
// src/app/dashboard/settings/page.tsx
// FIXED: Removed Stripe Connect (not available in SA without a US entity).
// FIXED: Added SA Bank Account section using the existing /api/payouts/bank-accounts endpoint.
// FIXED: Paystack section retains connected status badge.
// FIXED: Added Plan Management section with Paystack upgrade flow.

import { useEffect, useRef, useState } from 'react';
import {
  Loader2, CheckCircle2, Mic2, ExternalLink,
  QrCode, Download, Building2, Plus, Trash2, Wallet,
  Crown, Zap, Star, Check, ArrowRight, AlertTriangle,
} from 'lucide-react';

const PLAN_DEFS = [
  {
    slug: 'free', name: 'Free', priceZAR: 0, artistSharePct: 85, platformFeePct: 15,
    features: ['Up to 2 releases/month', 'Beat store & licensing', 'Fan memberships', 'Basic analytics'],
    color: 'var(--text-muted)', Icon: Zap,
  },
  {
    slug: 'pro', name: 'Pro', priceZAR: 249, artistSharePct: 92, platformFeePct: 8,
    features: ['Unlimited releases', 'Only 8% platform fee', 'Priority support', 'Advanced analytics', 'Industry marketplace'],
    color: 'var(--sky)', Icon: Crown,
  },
  {
    slug: 'label', name: 'Label', priceZAR: 999, artistSharePct: 95, platformFeePct: 5,
    features: ['Unlimited releases', 'Lowest 5% platform fee', 'Multi-artist management', 'Bulk payouts', 'White-label storefront'],
    color: 'var(--gold)', Icon: Star,
  },
];

const SA_BANKS = [
  { name: 'Absa', branch: '632005' },
  { name: 'Capitec', branch: '470010' },
  { name: 'FNB / First National Bank', branch: '250655' },
  { name: 'Nedbank', branch: '198765' },
  { name: 'Standard Bank', branch: '051001' },
  { name: 'African Bank', branch: '430000' },
  { name: 'Discovery Bank', branch: '679000' },
  { name: 'TymeBank', branch: '678910' },
  { name: 'Investec', branch: '580105' },
];

export default function SettingsPage() {
  const [artist, setArtist]             = useState<any>(null);
  const [loading, setLoading]           = useState(true);
  const [saving, setSaving]             = useState(false);
  const [saved, setSaved]               = useState(false);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [coverPreview, setCoverPreview] = useState<string | null>(null);
  const photoRef = useRef<HTMLInputElement>(null);
  const coverRef = useRef<HTMLInputElement>(null);

  // Bank accounts
  const [bankAccounts, setBankAccounts] = useState<any[]>([]);
  const [showBankForm, setShowBankForm] = useState(false);
  const [bankSaving, setBankSaving]     = useState(false);
  const [bankError, setBankError]       = useState<string | null>(null);
  const [bankForm, setBankForm]         = useState({
    accountHolder: '', bankName: '', branchCode: '', accountNumber: '',
  });

  // Plan management
  const [planInfo, setPlanInfo]         = useState<any>(null);
  const [planLoading, setPlanLoading]   = useState(false);
  const [cancellingPlan, setCancellingPlan] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch('/api/dashboard/settings').then(r => r.json()).then(d => setArtist(d.artist || d)),
      fetch('/api/payouts/bank-accounts').then(r => r.ok ? r.json() : { accounts: [] }).then(d => setBankAccounts(d.accounts || [])),
      fetch(`/api/plans/status?t=${Date.now()}`, { cache: 'no-store' }).then(r => r.ok ? r.json() : null).then(d => setPlanInfo(d)),
    ]).catch(() => {}).finally(() => setLoading(false));
  }, []);

  async function upgradePlan(planSlug: string) {
    setPlanLoading(true);
    try {
      const res = await fetch('/api/plans/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planSlug }),
      });
      const { authorizationUrl, error } = await res.json();
      if (error) { alert(error); return; }

      // Redirect to Paystack's hosted checkout page
      if (authorizationUrl) {
        window.location.href = authorizationUrl;
      }
    } catch {
      alert('Failed to start upgrade payment');
    }
    setPlanLoading(false);
  }

  async function cancelPlan() {
    if (!confirm('Cancel your plan? You\'ll keep access until the end of your billing period.')) return;
    setCancellingPlan(true);
    try {
      const res = await fetch('/api/plans/cancel', { method: 'POST' });
      const d = await res.json();
      if (d.ok) {
        alert(d.message);
        const updated = await fetch('/api/plans/status').then(r => r.json());
        setPlanInfo(updated);
      } else {
        alert(d.error || 'Failed to cancel plan');
      }
    } catch {
      alert('Failed to cancel plan');
    }
    setCancellingPlan(false);
  }

  async function pickImage(file: File, key: 'photoUrl' | 'coverUrl', setPreview: (s: string) => void) {
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    try {
      const type = key === 'coverUrl' ? 'cover' : 'photo';
      const mimeType = file.type || 'image/jpeg';
      const res = await fetch(`/api/dashboard/settings/upload-url?type=${type}&mimeType=${encodeURIComponent(mimeType)}`);
      if (!res.ok) throw new Error('Could not get upload URL');
      const { uploadUrl, publicUrl } = await res.json();
      const uploadRes = await fetch(uploadUrl, { method: 'PUT', body: file, headers: { 'Content-Type': mimeType } });
      if (!uploadRes.ok) throw new Error(`R2 upload failed: ${uploadRes.status}`);
      const updatedArtist = await new Promise<any>(resolve => {
        setArtist((p: any) => {
          const updated = { ...p, [key]: publicUrl };
          resolve(updated);
          return updated;
        });
      });
      await fetch('/api/dashboard/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: publicUrl }),
      });
    } catch (e: any) {
      console.error('Image upload failed:', e.message);
    }
  }

  async function saveProfile(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    await fetch('/api/dashboard/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(artist),
    }).catch(() => {});
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  async function saveBankAccount() {
    if (!bankForm.accountHolder || !bankForm.bankName || !bankForm.accountNumber) return;
    setBankSaving(true);
    setBankError(null);
    try {
      const res = await fetch('/api/payouts/bank-accounts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          accountType: 'current',
          accountHolder: bankForm.accountHolder,
          bankName: bankForm.bankName,
          branchCode: bankForm.branchCode,
          accountNumber: bankForm.accountNumber,
          isDefault: bankAccounts.length === 0,
        }),
      });
      const d = await res.json();
      if (res.ok) {
        setBankAccounts(prev => [...prev, d.account].filter(Boolean));
        setShowBankForm(false);
        setBankError(null);
        setBankForm({ accountHolder: '', bankName: '', branchCode: '', accountNumber: '' });
      } else {
        setBankError(d.error || 'Failed to save bank account. Please try again.');
      }
    } catch (err: any) {
      setBankError('Network error: ' + (err?.message || 'Please check your connection.'));
    }
    setBankSaving(false);
  }

  async function deleteBankAccount(id: string) {
    if (!confirm('Remove this bank account?')) return;
    try {
      await fetch(`/api/payouts/bank-accounts?id=${id}`, { method: 'DELETE' });
      setBankAccounts(prev => prev.filter(a => a.id !== id));
    } catch {}
  }

  if (loading) return (
    <div className="p-10 flex items-center gap-3" style={{ color: 'var(--text-muted)' }}>
      <Loader2 size={20} className="animate-spin" /> Loading your profile…
    </div>
  );
  if (!artist) return (
    <div className="p-10">
      <p style={{ color: 'var(--text-muted)' }}>You need an artist profile to access settings.</p>
    </div>
  );

  return (
    <div className="p-6 md:p-10 max-w-2xl">
      <h1 className="text-2xl font-black mb-2" style={{ color: 'var(--text)' }}>Settings</h1>
      <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
        Manage your profile, payments, and public store link.
      </p>

      {/* ── Payment Setup ── */}
      <div className="rounded-2xl mb-6" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="p-6 border-b" style={{ borderColor: 'var(--border)' }}>
          <h2 className="font-bold text-lg mb-1" style={{ color: 'var(--text)' }}>💳 Payment Setup</h2>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Add your bank account so Vuka can pay you every Friday via EFT.
          </p>
        </div>

        {/* SA Bank Account */}
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="font-semibold text-sm" style={{ color: 'var(--text)' }}>SA Bank Account — EFT Payouts</p>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                FNB, Absa, Standard Bank, Capitec, Nedbank and more
              </p>
            </div>
            <span className="text-xs px-2 py-1 rounded-full font-medium" style={{
              background: bankAccounts.length > 0 ? 'color-mix(in srgb,var(--green) 15%,transparent)' : 'var(--surface2)',
              color: bankAccounts.length > 0 ? 'var(--green)' : 'var(--text-muted)',
            }}>
              {bankAccounts.length > 0 ? `✓ ${bankAccounts.length} saved` : 'Not set up'}
            </span>
          </div>

          {/* Existing accounts */}
          {bankAccounts.length > 0 && (
            <div className="space-y-2 mb-4">
              {bankAccounts.map(a => (
                <div key={a.id} className="flex items-center gap-3 p-3 rounded-xl"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
                  <Building2 size={14} style={{ color: 'var(--gold)' }} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{a.bankName}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {a.accountHolder} · {a.maskedNumber || '****'}
                      {a.isDefault && <span className="ml-2 text-[10px] font-bold px-1.5 py-0.5 rounded"
                        style={{ background: 'rgba(16,185,129,0.1)', color: 'var(--green)' }}>DEFAULT</span>}
                    </p>
                  </div>
                  <button onClick={() => deleteBankAccount(a.id)}
                    className="p-1.5 rounded-lg"
                    style={{ color: 'var(--text-muted)' }}>
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {!showBankForm ? (
            <button onClick={() => { setShowBankForm(true); setBankError(null); }}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm"
              style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}>
              <Plus size={14} />
              {bankAccounts.length === 0 ? 'Add Bank Account' : 'Add Another Account'}
            </button>
          ) : (
            <div className="space-y-4 p-5 rounded-xl" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
              <p className="text-sm font-bold" style={{ color: 'var(--text)' }}>Add SA Bank Account</p>

              {/* Inline error banner */}
              {bankError && (
                <div className="flex items-start gap-2 p-3 rounded-lg text-sm"
                  style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)', color: '#f87171' }}>
                  <AlertTriangle size={15} className="mt-0.5 flex-shrink-0" />
                  <span>{bankError}</span>
                </div>
              )}

              <div>
                <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Account Holder Name</label>
                <input
                  value={bankForm.accountHolder}
                  onChange={e => setBankForm(p => ({ ...p, accountHolder: e.target.value }))}
                  placeholder="Full name as it appears on your bank account"
                  className="w-full px-4 py-3 rounded-xl text-sm"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                />
              </div>

              <div>
                <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Bank</label>
                <select
                  value={bankForm.bankName}
                  onChange={e => {
                    const bank = SA_BANKS.find(b => b.name === e.target.value);
                    setBankForm(p => ({ ...p, bankName: e.target.value, branchCode: bank?.branch || '' }));
                  }}
                  className="w-full px-4 py-3 rounded-xl text-sm"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                >
                  <option value="">Select your bank</option>
                  {SA_BANKS.map(b => (
                    <option key={b.name} value={b.name}>{b.name} (Branch: {b.branch})</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Branch Code</label>
                  <input
                    value={bankForm.branchCode}
                    onChange={e => setBankForm(p => ({ ...p, branchCode: e.target.value }))}
                    placeholder="Auto-filled"
                    className="w-full px-4 py-3 rounded-xl text-sm"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Account Number</label>
                  <input
                    value={bankForm.accountNumber}
                    onChange={e => setBankForm(p => ({ ...p, accountNumber: e.target.value }))}
                    placeholder="Your account number"
                    className="w-full px-4 py-3 rounded-xl text-sm"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
                  />
                </div>
              </div>

              <div className="flex gap-3 pt-1">
                <button onClick={saveBankAccount} disabled={bankSaving}
                  className="flex-1 py-3 rounded-xl font-bold text-sm text-white disabled:opacity-60"
                  style={{ background: 'var(--sky)' }}>
                  {bankSaving ? <><Loader2 size={14} className="animate-spin inline mr-2" />Saving…</> : 'Save Account'}
                </button>
                <button onClick={() => { setShowBankForm(false); setBankError(null); }}
                  className="px-5 py-3 rounded-xl text-sm font-medium"
                  style={{ background: 'var(--surface)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Currency Preference */}
      <div className="p-6 rounded-2xl mb-6" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h2 className="font-bold text-base mb-3" style={{ color: 'var(--text)' }}>Default Currency</h2>
        <select
          value={artist.currency || 'ZAR'}
          onChange={e => setArtist((p: any) => ({ ...p, currency: e.target.value }))}
          className="w-full px-4 py-3 rounded-xl text-sm"
          style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}>
          <option value="ZAR">ZAR — South African Rand</option>
          <option value="NGN">NGN — Nigerian Naira</option>
          <option value="KES">KES — Kenyan Shilling</option>
          <option value="GHS">GHS — Ghanaian Cedi</option>
          <option value="USD">USD — US Dollar</option>
          <option value="EUR">EUR — Euro</option>
          <option value="GBP">GBP — British Pound</option>
        </select>
      </div>

      {/* Profile Form */}
      <form onSubmit={saveProfile} className="space-y-5">
        <div className="flex items-center justify-between">
          <h2 className="font-bold text-lg" style={{ color: 'var(--text)' }}>Artist Profile</h2>
          <a href={`/artist/${artist.slug}`} target="_blank" rel="noopener noreferrer"
            className="text-xs flex items-center gap-1 underline"
            style={{ color: 'var(--sky)' }}>
            View public page <ExternalLink size={11} />
          </a>
        </div>

        {/* Photo Upload */}
        <div>
          <label className="block text-sm mb-2" style={{ color: 'var(--text-muted)' }}>Profile Photo</label>
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full overflow-hidden flex-shrink-0 flex items-center justify-center"
              style={{ background: 'var(--surface2)', border: '2px solid var(--border)' }}>
              {(photoPreview || artist.photoUrl)
                ? <img src={photoPreview || artist.photoUrl} alt="Photo" className="w-full h-full object-cover" />
                : <Mic2 size={24} style={{ color: 'var(--text-muted)' }} />}
            </div>
            <button type="button" onClick={() => photoRef.current?.click()}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}>
              Upload Photo
            </button>
            <input ref={photoRef} type="file" accept="image/*" className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) pickImage(f, 'photoUrl', setPhotoPreview); }} />
          </div>
        </div>

        {/* Cover Image */}
        <div>
          <label className="block text-sm mb-2" style={{ color: 'var(--text-muted)' }}>Cover / Banner Image</label>
          <div className="w-full h-28 rounded-xl overflow-hidden relative flex items-center justify-center cursor-pointer"
            style={{ background: 'var(--surface2)', border: '2px dashed var(--border)' }}
            onClick={() => coverRef.current?.click()}>
            {(coverPreview || artist.coverUrl)
              ? <img src={coverPreview || artist.coverUrl} alt="Cover" className="w-full h-full object-cover" />
              : <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Click to upload cover image</span>}
          </div>
          <input ref={coverRef} type="file" accept="image/*" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) pickImage(f, 'coverUrl', setCoverPreview); }} />
        </div>

        {/* Text Fields */}
        {[
          { label: 'Artist / Stage Name', key: 'name', placeholder: 'Your stage name' },
          { label: 'City', key: 'city', placeholder: 'e.g. Johannesburg' },
          { label: 'Country', key: 'country', placeholder: 'e.g. South Africa' },
        ].map(f => (
          <div key={f.key}>
            <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>{f.label}</label>
            <input value={artist[f.key] || ''} placeholder={f.placeholder}
              onChange={e => setArtist((p: any) => ({ ...p, [f.key]: e.target.value }))}
              className="w-full px-4 py-3 rounded-xl"
              style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
          </div>
        ))}

        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>Bio — tell fans your story</label>
          <textarea rows={4} value={artist.bio || ''} placeholder="Who are you? What's your sound? Where are you from?"
            onChange={e => setArtist((p: any) => ({ ...p, bio: e.target.value }))}
            className="w-full px-4 py-3 rounded-xl resize-none"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        </div>

        <div>
          <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>Genre Tags (comma separated)</label>
          <input value={(artist.genreTags || []).join(', ')}
            placeholder="e.g. Amapiano, Afrobeats, Hip Hop, Gqom"
            onChange={e => setArtist((p: any) => ({ ...p, genreTags: e.target.value.split(',').map((t: string) => t.trim()).filter(Boolean) }))}
            className="w-full px-4 py-3 rounded-xl"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        </div>

        {/* Social Links */}
        <div>
          <label className="block text-sm mb-2" style={{ color: 'var(--text-muted)' }}>Social Links</label>
          <div className="space-y-2">
            {['instagram', 'twitter', 'spotify', 'youtube', 'soundcloud'].map(platform => (
              <div key={platform} className="flex items-center gap-2">
                <span className="text-xs w-24 capitalize flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{platform}</span>
                <input
                  value={(artist.socialLinks as any)?.[platform] || ''}
                  placeholder={`https://${platform}.com/yourname`}
                  onChange={e => setArtist((p: any) => ({
                    ...p,
                    socialLinks: { ...(p.socialLinks || {}), [platform]: e.target.value }
                  }))}
                  className="flex-1 px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="pt-2">
          <button type="submit" disabled={saving}
            className="w-full py-4 rounded-xl font-bold text-white disabled:opacity-60 transition-all"
            style={{ background: saved ? 'var(--green)' : 'var(--sky)' }}>
            {saving
              ? <><Loader2 size={16} className="animate-spin inline mr-2" />Saving…</>
              : saved
              ? <><CheckCircle2 size={16} className="inline mr-2" />Profile Saved!</>
              : 'Save Profile'}
          </button>
        </div>
      </form>

      {/* ── PLAN MANAGEMENT ─────────────────────────────── */}
      <div id="plan" className="mt-6 p-6 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-1">
          <Crown size={18} style={{ color: 'var(--sky)' }} />
          <h2 className="font-bold text-base" style={{ color: 'var(--text)' }}>Your Plan</h2>
        </div>
        <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
          Upgrade to keep more of every sale and unlock unlimited releases.
        </p>

        <div className="grid grid-cols-1 gap-4 mb-4">
          {PLAN_DEFS.map(p => {
            const isActive = planInfo?.planSlug === p.slug;
            const Icon = p.Icon;
            return (
              <div key={p.slug} className="rounded-2xl p-6 flex flex-col"
                style={{
                  background: isActive ? `${p.color}0d` : 'var(--surface2)',
                  border: `1.5px solid ${isActive ? p.color : 'var(--border)'}`,
                }}>
                {/* Header row */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Icon size={18} style={{ color: p.color }} />
                    <span className="font-bold text-base" style={{ color: 'var(--text)' }}>{p.name}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {isActive && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                        style={{ background: `${p.color}22`, color: p.color }}>
                        ✓ Active
                      </span>
                    )}
                    <div>
                      {p.priceZAR === 0
                        ? <span className="text-xl font-black" style={{ color: p.color }}>Free</span>
                        : <><span className="text-xl font-black" style={{ color: p.color }}>R{p.priceZAR}</span>
                           <span className="text-sm" style={{ color: 'var(--text-muted)' }}>/mo</span></>
                      }
                    </div>
                  </div>
                </div>

                {/* Share line */}
                <p className="text-sm font-semibold mb-3" style={{ color: p.color }}>
                  You keep {p.artistSharePct}% · Vuka takes {p.platformFeePct}%
                </p>

                {/* Features */}
                <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 mb-4">
                  {p.features.map(f => (
                    <li key={f} className="flex items-start gap-1.5 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <Check size={13} className="mt-0.5 flex-shrink-0" style={{ color: 'var(--green)' }} />
                      {f}
                    </li>
                  ))}
                </ul>

                {/* Action */}
                {!isActive && p.priceZAR > 0 && (
                  <button
                    onClick={() => upgradePlan(p.slug)}
                    disabled={planLoading}
                    className="w-full py-3 rounded-xl text-sm font-bold text-white flex items-center justify-center gap-2 disabled:opacity-60"
                    style={{ background: p.color }}>
                    {planLoading ? <Loader2 size={14} className="animate-spin" /> : <ArrowRight size={14} />}
                    Upgrade to {p.name}
                  </button>
                )}
                {isActive && p.priceZAR > 0 && planInfo?.subscription?.status !== 'cancelled' && (
                  <button
                    onClick={cancelPlan}
                    disabled={cancellingPlan}
                    className="w-full py-3 rounded-xl text-sm font-medium disabled:opacity-60"
                    style={{ color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                    {cancellingPlan ? 'Cancelling…' : 'Cancel plan'}
                  </button>
                )}
                {isActive && planInfo?.subscription?.status === 'cancelled' && (
                  <div className="flex items-center gap-1.5 text-sm" style={{ color: 'var(--gold)' }}>
                    <AlertTriangle size={14} />
                    Access until {planInfo?.planExpiresAt ? new Date(planInfo.planExpiresAt).toLocaleDateString('en-ZA') : ''}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* QR Code */}
      <div className="mt-6 p-6 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-1">
          <QrCode size={18} style={{ color: 'var(--sky)' }} />
          <h2 className="font-bold text-base" style={{ color: 'var(--text)' }}>Your QR Code</h2>
        </div>
        <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
          Share or print this to let anyone scan and visit your store instantly.
        </p>
        <div className="flex items-center gap-6">
          <div className="w-28 h-28 rounded-xl overflow-hidden flex-shrink-0 flex items-center justify-center"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
            <img
              src="/api/dashboard/qr"
              alt="Your QR Code"
              className="w-full h-full object-cover"
              onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          </div>
          <div className="flex-1 space-y-3">
            <p className="text-xs font-mono truncate" style={{ color: 'var(--text-muted)' }}>
              {`${process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za'}/artist/${artist.slug}`}
            </p>
            <a
              href="/api/dashboard/qr"
              download={`vuka-qr-${artist.slug}.png`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm text-white"
              style={{ background: 'var(--sky)' }}>
              <Download size={14} />
              Download QR Code
            </a>
          </div>
        </div>
      </div>

    </div>
  );
}
