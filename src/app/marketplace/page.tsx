'use client';
// src/app/marketplace/page.tsx
// Public marketplace — fans and artists browse & order services (mixing, mastering, features, etc.)

import { useEffect, useState, useCallback } from 'react';
import Navbar from '@/components/Navbar';
import {
  Search, Loader2, Briefcase, Clock,
  Star, Music, Mic2, Video, Package,
  Sliders, ShoppingCart, X, AlertCircle,
} from 'lucide-react';

const CATEGORIES = [
  { value: '',             label: 'All Services',      icon: Briefcase },
  { value: 'mixing',       label: 'Mixing',            icon: Sliders },
  { value: 'mastering',    label: 'Mastering',         icon: Sliders },
  { value: 'features',     label: 'Features / Vocals', icon: Mic2 },
  { value: 'production',   label: 'Beat Production',   icon: Music },
  { value: 'ghostwriting', label: 'Ghostwriting',      icon: Package },
  { value: 'videography',  label: 'Videography',       icon: Video },
  { value: 'photography',  label: 'Photography',       icon: Package },
  { value: 'promotion',    label: 'Promotion',         icon: Package },
  { value: 'other',        label: 'Other',             icon: Package },
];

export default function MarketplacePage() {
  const [services, setServices] = useState<any[]>([]);
  const [loading, setLoading]   = useState(true);
  const [q, setQ]               = useState('');
  const [search, setSearch]     = useState('');
  const [category, setCategory] = useState('');
  const [page, setPage]         = useState(1);
  const [total, setTotal]       = useState(0);
  const [pages, setPages]       = useState(1);

  // Checkout modal state
  const [ordering, setOrdering]         = useState<any | null>(null);
  const [selectedPkg, setSelectedPkg]   = useState<any | null>(null);
  const [requirements, setRequirements] = useState('');
  const [buyerName, setBuyerName]       = useState('');
  const [buyerEmail, setBuyerEmail]     = useState('');
  const [checkoutErr, setCheckoutErr]   = useState('');
  const [paying, setPaying]             = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams();
    if (search)   params.set('search', search);
    if (category) params.set('category', category);
    params.set('take', '20');
    params.set('skip', String((page - 1) * 20));
    fetch(`/api/marketplace/services?${params}`)
      .then(r => r.ok ? r.json() : { services: [] })
      .then(d => {
        setServices(d.services || []);
        setTotal(d.total || 0);
        setPages(d.pages || 1);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [search, category, page]);

  useEffect(() => { load(); }, [load]);

  function handleOrder(service: any) {
    const lowestPkg = service.packages?.reduce(
      (min: any, p: any) => (!min || p.price < min.price ? p : min), null
    );
    setOrdering(service);
    setSelectedPkg(lowestPkg || null);
    setRequirements('');
    setBuyerName('');
    setBuyerEmail('');
    setCheckoutErr('');
  }

  async function handleCheckout() {
    if (!ordering) return;
    if (!buyerName.trim() || !buyerEmail.trim()) {
      setCheckoutErr('Your name and email are required.');
      return;
    }
    if (!selectedPkg) {
      setCheckoutErr('Select a package to continue.');
      return;
    }
    setPaying(true);
    setCheckoutErr('');
    try {
      const res = await fetch('/api/marketplace/checkout', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          serviceId:    ordering.id,
          packageName:  selectedPkg.name,
          amount:       selectedPkg.price,
          requirements,
          buyerName:    buyerName.trim(),
          buyerEmail:   buyerEmail.trim(),
        }),
      });
      const data = await res.json();
      if (!res.ok) { setCheckoutErr(data.error || 'Could not start payment.'); setPaying(false); return; }

      // Redirect to Paystack's hosted checkout page
      if (data.authorizationUrl) {
        window.location.href = data.authorizationUrl;
      } else {
        setCheckoutErr('Payment gateway not configured');
        setPaying(false);
      }
    } catch {
      setCheckoutErr('Network error. Please try again.');
      setPaying(false);
    }
  }

  return (
    <>
      <Navbar />
      <main className="min-h-screen" style={{ background: 'var(--bg)', color: 'var(--text)' }}>
        <div className="max-w-6xl mx-auto px-4 py-10">

          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-black font-display mb-2">Marketplace</h1>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Hire artists for mixing, mastering, features, production, and more.
            </p>
          </div>

          {/* Search + filter */}
          <div className="flex flex-wrap gap-3 mb-6">
            <div className="flex-1 min-w-[200px] relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
              <input
                value={q}
                onChange={e => setQ(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') { setSearch(q); setPage(1); } }}
                placeholder="Search services…"
                className="w-full pl-8 pr-4 py-2.5 rounded-xl text-sm outline-none"
                style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }}
              />
            </div>
            <button
              onClick={() => { setSearch(q); setPage(1); }}
              className="px-4 py-2.5 rounded-xl text-sm font-semibold"
              style={{ background: 'var(--green)', color: '#0a0a0a' }}>
              Search
            </button>
          </div>

          {/* Category pills */}
          <div className="flex gap-2 flex-wrap mb-8">
            {CATEGORIES.map(cat => (
              <button
                key={cat.value}
                onClick={() => { setCategory(cat.value); setPage(1); }}
                className="px-3 py-1.5 rounded-full text-xs font-semibold transition-all"
                style={{
                  background:  category === cat.value ? 'var(--green)' : 'var(--surface)',
                  color:       category === cat.value ? '#0a0a0a' : 'var(--text-muted)',
                  border:      category === cat.value ? '1px solid var(--green)' : '1px solid var(--border)',
                }}>
                {cat.label}
              </button>
            ))}
          </div>

          {/* Results count */}
          <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
            {loading ? 'Loading…' : `${total} service${total !== 1 ? 's' : ''} available`}
          </p>

          {/* Services grid */}
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="animate-spin" size={28} style={{ color: 'var(--green)' }} />
            </div>
          ) : !services.length ? (
            <div className="text-center py-20" style={{ color: 'var(--text-muted)' }}>
              <Briefcase size={40} className="mx-auto mb-4 opacity-40" />
              <p className="font-semibold">No services found</p>
              <p className="text-sm mt-1">Try a different category or search term</p>
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {services.map((svc: any) => (
                <ServiceCard key={svc.id} service={svc} onOrder={handleOrder} />
              ))}
            </div>
          )}

          {/* Pagination */}
          {pages > 1 && (
            <div className="flex gap-2 mt-8 justify-center flex-wrap">
              {Array.from({ length: pages }, (_, i) => (
                <button key={i} onClick={() => setPage(i + 1)}
                  className="w-9 h-9 rounded-xl text-sm font-mono"
                  style={{
                    background: page === i + 1 ? 'var(--green)' : 'var(--surface)',
                    color:      page === i + 1 ? '#0a0a0a' : 'var(--text)',
                    border:     '1px solid var(--border)',
                  }}>
                  {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Checkout modal */}
      {ordering && (
        <>
          <div className="fixed inset-0 z-40 bg-black/60" onClick={() => { if (!paying) setOrdering(null); }} />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="w-full max-w-md rounded-2xl p-6 overflow-y-auto max-h-[90vh]"
              style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

              <div className="flex items-start justify-between mb-5">
                <div>
                  <p className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: 'var(--text-muted)' }}>Order Service</p>
                  <h2 className="text-lg font-bold" style={{ color: 'var(--text)' }}>{ordering.title}</h2>
                  <p className="text-xs mt-0.5" style={{ color: 'var(--green)' }}>{ordering.artist?.name}</p>
                </div>
                {!paying && (
                  <button onClick={() => setOrdering(null)} style={{ color: 'var(--text-muted)' }}>
                    <X size={18} />
                  </button>
                )}
              </div>

              {/* Package selector */}
              {ordering.packages?.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Select package</p>
                  <div className="space-y-2">
                    {ordering.packages.map((pkg: any) => (
                      <button key={pkg.name} onClick={() => setSelectedPkg(pkg)}
                        className="w-full text-left p-3 rounded-xl transition-all"
                        style={{
                          background: selectedPkg?.name === pkg.name ? 'var(--surface2)' : 'transparent',
                          border:     `1px solid ${selectedPkg?.name === pkg.name ? 'var(--green)' : 'var(--border)'}`,
                        }}>
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{pkg.name}</span>
                          <span className="text-sm font-black font-mono" style={{ color: 'var(--green)' }}>R{pkg.price}</span>
                        </div>
                        {pkg.description && <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{pkg.description}</p>}
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{pkg.deliveryDays}d delivery</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Requirements */}
              <div className="mb-4">
                <label className="block text-xs font-semibold mb-1.5" style={{ color: 'var(--text-muted)' }}>
                  Project requirements <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span>
                </label>
                <textarea
                  value={requirements}
                  onChange={e => setRequirements(e.target.value)}
                  placeholder="Describe what you need, share links, references…"
                  rows={3}
                  className="w-full px-3 py-2 rounded-xl text-sm resize-none outline-none"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                />
              </div>

              {/* Buyer info */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                  <label className="block text-xs font-semibold mb-1.5" style={{ color: 'var(--text-muted)' }}>Your name *</label>
                  <input value={buyerName} onChange={e => setBuyerName(e.target.value)}
                    placeholder="Full name"
                    className="w-full px-3 py-2 rounded-xl text-sm outline-none"
                    style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1.5" style={{ color: 'var(--text-muted)' }}>Email *</label>
                  <input type="email" value={buyerEmail} onChange={e => setBuyerEmail(e.target.value)}
                    placeholder="you@email.com"
                    className="w-full px-3 py-2 rounded-xl text-sm outline-none"
                    style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }} />
                </div>
              </div>

              {checkoutErr && (
                <div className="flex items-center gap-2 p-3 rounded-xl text-xs mb-4"
                  style={{ background: 'rgba(232,64,64,0.1)', border: '1px solid rgba(232,64,64,0.3)', color: '#f87171' }}>
                  <AlertCircle size={14} className="flex-shrink-0" />
                  {checkoutErr}
                </div>
              )}

              <button onClick={handleCheckout} disabled={paying}
                className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 disabled:opacity-60"
                style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                {paying
                  ? <><Loader2 size={15} className="animate-spin" /> Redirecting to payment…</>
                  : <><ShoppingCart size={15} /> Pay R{selectedPkg?.price ?? 0} — Secure checkout</>
                }
              </button>
              <p className="text-center text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                Powered by Paystack · Work begins after payment confirms
              </p>
            </div>
          </div>
        </>
      )}
    </>
  );
}

function ServiceCard({ service, onOrder }: { service: any; onOrder: (s: any) => void }) {
  const lowestPkg = service.packages?.reduce(
    (min: any, p: any) => (!min || p.price < min.price ? p : min),
    null
  );
  const price = lowestPkg?.price ?? service.price ?? 0;

  return (
    <div
      className="rounded-2xl overflow-hidden flex flex-col cursor-pointer hover:border-white/20 transition-all"
      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
      onClick={() => onOrder(service)}>

      {/* Top color band */}
      <div className="h-1.5" style={{ background: 'var(--green)' }} />

      <div className="p-5 flex flex-col flex-1 gap-3">
        {/* Artist */}
        <div className="flex items-center gap-2">
          {service.artist?.photoUrl
            ? <img src={service.artist.photoUrl} className="w-8 h-8 rounded-full object-cover flex-shrink-0" />
            : <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center"
                style={{ background: 'var(--surface2)' }}>
                <Mic2 size={12} style={{ color: 'var(--text-muted)' }} />
              </div>
          }
          <div>
            <p className="text-xs font-semibold" style={{ color: 'var(--green)' }}>{service.artist?.name}</p>
            {service.artist?.isVerified && (
              <span className="text-[10px] font-bold" style={{ color: 'var(--sky)' }}>✓ Verified</span>
            )}
          </div>
        </div>

        {/* Title */}
        <div>
          <h3 className="font-bold text-sm leading-snug">{service.title}</h3>
          <p className="text-xs mt-1 line-clamp-2" style={{ color: 'var(--text-muted)' }}>{service.description}</p>
        </div>

        {/* Category + delivery */}
        <div className="flex gap-2 flex-wrap">
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full capitalize"
            style={{ background: 'rgba(160,232,124,0.12)', color: 'var(--green)' }}>
            {service.category}
          </span>
          {service.deliveryDays && (
            <span className="text-[10px] flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
              <Clock size={10} /> {service.deliveryDays}d delivery
            </span>
          )}
          {service._count?.reviews > 0 && (
            <span className="text-[10px] flex items-center gap-1" style={{ color: '#e8c87c' }}>
              <Star size={10} fill="currentColor" /> {service._count.reviews} reviews
            </span>
          )}
        </div>

        {/* Price + order */}
        <div className="flex items-center justify-between mt-auto pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
          <div>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Starting from</p>
            <p className="text-lg font-black font-mono" style={{ color: 'var(--green)' }}>
              R{price.toFixed(0)}
            </p>
          </div>
          <button
            onClick={e => { e.stopPropagation(); onOrder(service); }}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold"
            style={{ background: 'var(--green)', color: '#0a0a0a' }}>
            <ShoppingCart size={12} /> Order
          </button>
        </div>
      </div>
    </div>
  );
}
