'use client';
// src/components/BuyModal.tsx
// Phase 12 — Paystack-only checkout (Stripe removed, PayFast replaced by Paystack)
// All purchases route to /api/checkout/paystack/initialize
// Paystack handles ZAR card, instant EFT, bank transfer, and mobile money

import { useState, useEffect } from 'react';
import { formatCurrency } from '@/lib/utils';
import { createClient } from '@/lib/supabase';

const LICENSES = [
  {
    key: 'basic',
    name: 'Basic License',
    rights: ['Non-exclusive', 'Up to 5,000 streams', '2 music videos', 'Non-profit performances'],
  },
  {
    key: 'premium',
    name: 'Premium License',
    rights: ['Non-exclusive', 'Up to 500K streams', 'Unlimited videos', 'Commercial performances', 'Radio rights'],
  },
  {
    key: 'exclusive',
    name: 'Exclusive License',
    rights: ['EXCLUSIVE — no one else can buy', 'Unlimited streams', 'Full commercial use', 'Sync/TV/Film', '50% songwriter credit'],
    highlight: true,
  },
];

interface Beat {
  id: string; title: string; artworkUrl: string;
  basicPrice: number; premiumPrice: number; exclPrice: number;
  artistSharePct?: number;
  artist: { name: string };
}

interface BuyModalProps {
  beat?: Beat;
  release?: {
    id: string; title: string; artworkUrl: string;
    price: number; minPrice: number; payWhatWant: boolean;
    artistSharePct?: number;
    artist: { name: string };
  };
  itemType?: string;  // override for video/sample; defaults to 'beat' or 'release'
  onClose: () => void;
}

export function BuyModal({ beat, release, itemType: itemTypeProp, onClose }: BuyModalProps) {
  const [license, setLicense]           = useState('basic');
  const [email, setEmail]               = useState('');
  const [name, setName]                 = useState('');
  const [customAmount, setCustomAmount] = useState('');
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState('');
  const [loggedInUserId, setLoggedInUserId] = useState<string | null>(null);

  // Auto-fill from logged-in session if available
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (!data.user) return;
      // Get full name + email from /api/auth/me (has DB name, not just auth metadata)
      fetch('/api/auth/me').then(r => r.ok ? r.json() : null).then(me => {
        if (!me) return;
        setLoggedInUserId(me.id ?? null);
        if (me.name && !name) setName(me.name);
        if (me.email && !email) setEmail(me.email);
      }).catch(() => {});
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const prices: Record<string, number> = beat
    ? { basic: beat.basicPrice, premium: beat.premiumPrice, exclusive: beat.exclPrice }
    : {};
  const price = beat
    ? prices[license]
    : (parseFloat(customAmount) || release!.price);

  async function handleBuy() {
    if (!email || !name) { setError('Please enter your name and email'); return; }
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/checkout/paystack/initialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemType:     itemTypeProp ?? (beat ? 'beat' : 'release'),
          itemId:       beat ? beat.id : release!.id,
          licenseType:  beat ? license : undefined,
          customAmount: release?.payWhatWant ? parseFloat(customAmount) : undefined,
          buyerEmail:   email,
          buyerName:    name,
          currency:     'ZAR',
          userId:       loggedInUserId ?? undefined,  // links purchase to account
        }),
      });

      const data = await res.json();
      if (!res.ok) { setError(data.error || 'Checkout error'); return; }

      // Free item — redirect directly to success page
      if (data.method === 'free') {
        window.location.href = data.url;
        return;
      }

      // Paid item — redirect user to Paystack's hosted checkout page
      if (data.authorizationUrl) {
        window.location.href = data.authorizationUrl;
        return;
      }

      setError('Payment gateway not configured');

    } catch {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  const item = beat || release!;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4"
      style={{ background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(6px)' }}
      onClick={onClose}
    >
      <div
        className="w-full sm:max-w-md rounded-t-2xl sm:rounded-2xl p-5 sm:p-6 max-h-[92svh] overflow-y-auto"
        style={{
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border-strong)',
          boxShadow: '0 24px 80px rgba(0,0,0,0.6)',
          paddingBottom: 'max(20px, env(safe-area-inset-bottom))',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start gap-4 mb-6">
          {item.artworkUrl ? (
            <img src={item.artworkUrl} className="w-16 h-16 rounded-lg object-cover" alt="" />
          ) : (
            <div
              className="w-16 h-16 rounded-lg flex items-center justify-center text-2xl"
              style={{ background: 'var(--color-bg-tertiary)' }}
            >🎵</div>
          )}
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-lg truncate" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
              {item.title}
            </h3>
            <p className="text-sm truncate" style={{ color: 'var(--color-text-secondary)' }}>
              {item.artist.name}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-2xl leading-none flex-shrink-0 hover:opacity-70 transition-opacity"
            style={{ color: 'var(--color-text-secondary)' }}
          >×</button>
        </div>

        {/* License picker (beats only) */}
        {beat && (
          <div className="mb-6 space-y-2">
            {LICENSES.map((l) => (
              <button
                key={l.key}
                onClick={() => setLicense(l.key)}
                className="w-full p-4 rounded-lg text-left transition-all"
                style={{
                  background: license === l.key ? 'rgba(160,232,124,0.08)' : 'var(--color-bg-tertiary)',
                  border: `2px solid ${license === l.key ? 'var(--color-accent-green)' : 'var(--color-border)'}`,
                }}
              >
                <div className="flex justify-between items-center">
                  <span className="font-bold" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
                    {l.name}
                  </span>
                  <span className="font-bold font-mono" style={{ color: 'var(--color-accent-green)' }}>
                    {formatCurrency(prices[l.key])}
                  </span>
                </div>
                <ul className="mt-1">
                  {l.rights.map((r) => (
                    <li key={r} className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>· {r}</li>
                  ))}
                </ul>
              </button>
            ))}
          </div>
        )}

        {/* Pay what you want (releases) */}
        {release?.payWhatWant && (
          <div className="mb-4">
            <label className="text-sm mb-1 block" style={{ color: 'var(--color-text-secondary)' }}>
              Your price (min R{release.minPrice})
            </label>
            <input
              type="number"
              value={customAmount}
              onChange={e => setCustomAmount(e.target.value)}
              placeholder={String(release.price)}
              className="input"
            />
          </div>
        )}

        {/* Buyer info */}
        <div className="space-y-3 mb-6">
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Your name"
            className="input"
          />
          <input
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="Email address (for download link)"
            className="input"
          />
        </div>

        {/* Paystack badge */}
        {price > 0 && (
          <div
            className="mb-4 px-3 py-2 rounded-lg flex items-center gap-2"
            style={{ background: 'rgba(160,232,124,0.07)', border: '1px solid rgba(160,232,124,0.2)' }}
          >
            <span style={{ fontSize: 18 }}>🇿🇦</span>
            <p style={{ color: 'var(--color-text-secondary)', fontSize: 12 }}>
              Secure payment via <strong style={{ color: 'var(--color-text-primary)' }}>Paystack</strong> — 
              card, instant EFT, bank transfer &amp; more accepted.
            </p>
          </div>
        )}

        {/* Platform fee note */}
        {price > 0 && (() => {
          const share = beat?.artistSharePct ?? release?.artistSharePct ?? 85;
          const fee   = 100 - share;
          return (
            <div
              className="mb-4 px-3 py-2 rounded-lg"
              style={{ background: 'rgba(232,200,124,0.07)', border: '1px solid rgba(232,200,124,0.2)' }}
            >
              <p style={{ color: 'var(--color-text-secondary)', fontSize: 11 }}>
                ✦ Vuka takes {fee}% to keep the platform running. The artist receives {share}% of this sale.
              </p>
            </div>
          );
        })()}

        {/* Price summary */}
        <div
          className="mb-5 p-4 rounded-lg"
          style={{ background: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border-strong)' }}
        >
          <div className="flex justify-between font-bold text-lg">
            <span style={{ color: 'var(--color-text-primary)' }}>Total</span>
            <span className="font-mono" style={{ color: 'var(--color-accent-green)' }}>
              {price === 0 ? 'Free' : formatCurrency(price)}
            </span>
          </div>
        </div>

        {error && (
          <div
            className="mb-4 p-3 rounded-lg text-sm"
            style={{ background: 'rgba(255,77,77,0.1)', border: '1px solid rgba(255,77,77,0.25)', color: 'var(--color-danger)' }}
          >
            {error}
          </div>
        )}

        <button
          onClick={handleBuy}
          disabled={loading}
          className="w-full py-4 rounded-lg font-bold text-base transition-all disabled:opacity-60"
          style={{
            background: price === 0 ? 'var(--color-accent-green)' : 'var(--color-accent-green)',
            color: '#000',
            fontFamily: 'var(--font-display)',
          }}
        >
          {loading
            ? 'Processing…'
            : price === 0
            ? 'Download Free →'
            : `Buy via Paystack — ${formatCurrency(price)} →`}
        </button>
      </div>
    </div>
  );
}
