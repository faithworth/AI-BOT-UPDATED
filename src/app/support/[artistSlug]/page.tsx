'use client';
import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import { formatCurrency, getTierFromAmount } from '@/lib/utils';

const QUICK_AMOUNTS = [20, 50, 100, 200, 500];
const TIERS = [
  { name: 'Listener', min: 0, icon: '👂' },
  { name: 'Supporter', min: 50, icon: '🤝' },
  { name: 'Day One', min: 200, icon: '⭐' },
  { name: 'Ride or Die', min: 500, icon: '🔥' },
];

export default function SupportPage() {
  const { artistSlug } = useParams<{ artistSlug: string }>();
  const [artist, setArtist] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [amount, setAmount] = useState(50);
  const [customAmount, setCustomAmount] = useState('');
  const [message, setMessage] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`/api/artist/${artistSlug}/profile`).then(r => r.json()).then(d => { setArtist(d); setLoading(false); }).catch(() => setLoading(false));
  }, [artistSlug]);

  const effectiveAmount = customAmount ? parseFloat(customAmount) || 0 : amount;
  const tier = getTierFromAmount(effectiveAmount);

  async function handleSupport() {
    if (!name || !email) { setError('Name and email required'); return; }
    if (effectiveAmount < 5) { setError('Minimum support is R5'); return; }
    setSubmitting(true); setError('');
    try {
      const res = await fetch('/api/support/create-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artistSlug, amount: effectiveAmount, message, fanName: name, fanEmail: email, isPublic, tier }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || 'Error'); return; }
      if (data.authorizationUrl) {
        window.location.href = data.authorizationUrl;
      } else {
        setError(data.error || 'Payment gateway not configured');
      }
    } catch { setError('Network error'); }
    finally { setSubmitting(false); }
  }

  if (loading) return <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg)' }}><p style={{ color: 'var(--text-muted)' }}>Just now…</p></div>;
  if (!artist) return <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg)' }}><p style={{ color: 'var(--text-muted)' }}>Eish. Artist not found.</p></div>;

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <Navbar />
      <div className="max-w-lg mx-auto px-4 py-12">
        {/* Artist header */}
        <div className="text-center mb-8">
          <div className="w-20 h-20 rounded-full overflow-hidden mx-auto mb-4" style={{ background: 'var(--surface2)' }}>
            {artist.photoUrl ? <img src={artist.photoUrl} alt={artist.name} className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-3xl">🎤</div>}
          </div>
          <h1 className="text-2xl font-black" style={{ color: 'var(--text)' }}>Support {artist.name} ♥</h1>
          <p style={{ color: 'var(--text-muted)' }}>{artist.city}</p>
        </div>

        {/* Tier guide */}
        <div className="grid grid-cols-4 gap-2 mb-8">
          {TIERS.map(t => (
            <div key={t.name} className="text-center p-3 rounded-xl" style={{ background: 'var(--surface)', border: `1px solid ${tier === t.name ? 'var(--sky)' : 'var(--border)'}` }}>
              <div className="text-xl">{t.icon}</div>
              <div className="text-xs font-bold mt-1" style={{ color: tier === t.name ? 'var(--sky)' : 'var(--text-muted)' }}>{t.name}</div>
              {t.min > 0 && <div className="text-xs" style={{ color: 'var(--text-muted)' }}>R{t.min}+</div>}
            </div>
          ))}
        </div>

        <div className="p-6 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
          {/* Quick amounts */}
          <div className="flex gap-2 mb-4 flex-wrap">
            {QUICK_AMOUNTS.map(a => (
              <button key={a} onClick={() => { setAmount(a); setCustomAmount(''); }}
                className="flex-1 py-2 rounded-lg font-bold text-sm transition-colors"
                style={{ background: amount === a && !customAmount ? 'var(--sky)' : 'var(--surface2)', border: '1px solid var(--border)', color: amount === a && !customAmount ? 'white' : 'var(--text-muted)', minWidth: 60 }}>
                R{a}
              </button>
            ))}
          </div>
          <div className="mb-4">
            <input type="number" value={customAmount} onChange={e => { setCustomAmount(e.target.value); setAmount(0); }} placeholder="Custom amount"
              className="w-full px-4 py-3 rounded-xl" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
          </div>

          {/* Current tier badge */}
          <div className="flex items-center justify-center gap-2 mb-6 p-3 rounded-xl" style={{ background: 'var(--surface2)' }}>
            <span>{TIERS.find(t => t.name === tier)?.icon}</span>
            <span className="font-bold" style={{ color: 'var(--sky)' }}>{tier} tier · {formatCurrency(effectiveAmount)}</span>
          </div>

          {/* Buyer info */}
          <div className="space-y-3 mb-4">
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Your name"
              className="w-full px-4 py-3 rounded-xl" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="Your email"
              className="w-full px-4 py-3 rounded-xl" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
            <textarea value={message} onChange={e => setMessage(e.target.value)} placeholder="Leave a message (optional)" rows={3}
              className="w-full px-4 py-3 rounded-xl resize-none" style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
          </div>

          <div className="flex items-center gap-3 mb-6">
            <input type="checkbox" id="pub" checked={isPublic} onChange={e => setIsPublic(e.target.checked)} />
            <label htmlFor="pub" className="text-sm" style={{ color: 'var(--text-muted)' }}>Show my support on the public wall</label>
          </div>

          {error && <p className="text-sm text-red-400 mb-4">Eish — {error}</p>}

          <button onClick={handleSupport} disabled={submitting || effectiveAmount < 5}
            className="w-full py-4 rounded-xl font-bold text-white text-lg disabled:opacity-60"
            style={{ background: 'linear-gradient(135deg,#f59e0b,#d97706)' }}>
            {submitting ? 'Just now…' : `♥ Support ${artist.name} · ${formatCurrency(effectiveAmount)}`}
          </button>
        </div>
      </div>
    </div>
  );
}
