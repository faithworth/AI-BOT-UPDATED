'use client';
import Link from 'next/link';
import { ArrowRight, Globe, Zap, DollarSign, Shield, Music, TrendingUp, Users, Star, Headphones } from 'lucide-react';
import Navbar from '@/components/Navbar';

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main style={{ background: 'var(--bg)', color: 'var(--text)' }}>

        {/* JSON-LD structured data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebSite",
              "name": "Vuka",
              "alternateName": ["Vuka Music", "Vuka Distro"],
              "url": "https://www.vuka.co.za",
              "description": "Africa's independent music platform. Buy beats and music directly from African artists.",
              "potentialAction": {
                "@type": "SearchAction",
                "target": "https://www.vuka.co.za/store?q={search_term_string}",
                "query-input": "required name=search_term_string"
              }
            })
          }}
        />

        {/* ── HERO ── */}
        <section className="relative min-h-[100svh] flex items-center justify-center overflow-hidden pt-16 pb-8">
          <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(56,182,232,0.10) 0%, transparent 70%)' }} />
          <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse 40% 30% at 80% 70%, rgba(201,162,39,0.07) 0%, transparent 60%)' }} />

          <div className="relative z-10 text-center px-4 max-w-5xl mx-auto w-full">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold mb-6"
              style={{ background: 'rgba(56,182,232,0.1)', border: '1px solid rgba(56,182,232,0.25)', color: 'var(--sky)' }}>
              <span className="w-1.5 h-1.5 rounded-full animate-pulse inline-block" style={{ background: 'var(--sky)' }} />
              Africa's independent music platform
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold mb-6 tracking-tight" style={{ color: 'var(--text)', lineHeight: 1.05 }}>
              Your music.<br />
              <span style={{
                background: 'linear-gradient(135deg, #38b6e8, #1a9dd4, #c9a227)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}>Your terms.</span><br />
              Your money.
            </h1>

            <p className="text-base sm:text-lg md:text-xl mb-4 max-w-2xl mx-auto leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Sell beats and releases directly to your fans — in South Africa and worldwide.
              Keep up to 95% of every sale.
            </p>

            <p className="text-sm mb-4 max-w-xl mx-auto" style={{ color: 'var(--text-muted)' }}>
              Paystack for South African buyers. Money goes directly to your bank. Start free — upgrade anytime for a lower platform fee.
            </p>

            {/* Fee transparency notice */}
            <div className="inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs mb-8"
              style={{ background: 'rgba(201,162,39,0.08)', border: '1px solid rgba(201,162,39,0.3)', color: 'var(--text-muted)' }}>
              <span style={{ color: 'var(--gold)' }}>✦</span>
              Free plan: 15% platform fee. Pro plan: 8%. Label plan: 5%. No hidden charges.
            </div>

            <div className="flex flex-col sm:flex-row gap-3 justify-center mb-10 px-2">
              <Link href="/auth/register" className="btn btn-primary text-sm sm:text-base px-6 py-3.5 w-full sm:w-auto">
                Start Selling — It's Free <ArrowRight size={16} />
              </Link>
              <Link href="/store" className="btn btn-secondary text-sm sm:text-base px-6 py-3.5 w-full sm:w-auto">
                Browse the Store <Music size={16} />
              </Link>
            </div>

            <div className="flex items-center justify-center gap-8 sm:gap-12 flex-wrap">
              {[
                { n: '85%', l: 'Artist keeps (Free)', sub: 'up to 95% on paid plans' },
                { n: '15%', l: 'Platform Fee', sub: 'Free plan — covers hosting & payments' },
                { n: 'ZAR', l: 'Paid in Rands' },
              ].map(s => (
                <div key={s.n} className="text-center">
                  <div className="text-2xl md:text-3xl font-bold mb-1" style={{ color: 'var(--text)' }}>{s.n}</div>
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.l}</div>
                  {'sub' in s && s.sub && <div className="text-xs mt-0.5" style={{ color: 'var(--gold)', fontSize: 10 }}>{s.sub}</div>}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── FOR ARTISTS ── */}
        <section className="py-24 px-4" style={{ background: 'var(--surface)' }}>
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-16">
              <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--sky)' }}>For Artists & Producers</p>
              <h2 className="text-3xl md:text-5xl font-bold mb-4" style={{ color: 'var(--text)' }}>
                Everything you need to get paid
              </h2>
              <p className="max-w-xl mx-auto" style={{ color: 'var(--text-muted)' }}>
                Upload your music in minutes. Set your price. Fans buy directly. Start on Free — upgrade to keep more.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
              {[
                {
                  icon: DollarSign,
                  title: 'Dual Payments',
                  desc: 'Paystack for South African buyers with instant EFT, card, and bank transfer support. Flutterwave for Pan-African payments. Both fully automated.',
                },
                {
                  icon: Zap,
                  title: 'Instant Downloads',
                  desc: 'Fans receive secure download links the moment payment clears. No manual work needed.',
                },
                {
                  icon: Shield,
                  title: 'Beat Licensing',
                  desc: 'Basic, Premium, and Exclusive tiers. Auto-generated PDF license agreements sent to every buyer.',
                },
                {
                  icon: Users,
                  title: 'Fan Support',
                  desc: 'Let fans tip you and back your recording goals. Build a real community around your music.',
                },
              ].map(f => (
                <div key={f.title} className="p-6 rounded-2xl transition-all duration-200"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--sky)';
                    (e.currentTarget as HTMLDivElement).style.boxShadow = '0 4px 20px rgba(56,182,232,0.1)';
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border)';
                    (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
                  }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4" style={{ background: 'rgba(56,182,232,0.12)' }}>
                    <f.icon size={20} style={{ color: 'var(--sky)' }} />
                  </div>
                  <h3 className="font-semibold mb-2" style={{ color: 'var(--text)' }}>{f.title}</h3>
                  <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="py-24 px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-5xl font-bold mb-4" style={{ color: 'var(--text)' }}>
                From studio to sold — in minutes
              </h2>
            </div>
            <div className="grid md:grid-cols-3 gap-10">
              {[
                { n: '01', t: 'Upload your music', d: 'Add your beats or releases. Set your prices, license tiers, and artwork. We handle the rest.' },
                { n: '02', t: 'Share your link', d: 'Get your personal store link — vuka.co.za/artist/you. Share it everywhere you already are.' },
                { n: '03', t: 'Get paid 98%', d: 'Paystack and Flutterwave move money directly to your bank account. Vuka retains just 2% to keep the platform running.' },
              ].map(s => (
                <div key={s.n} className="text-center">
                  <div className="text-5xl font-bold mb-4 font-mono" style={{ color: 'var(--border)' }}>{s.n}</div>
                  <h3 className="font-semibold text-lg mb-3" style={{ color: 'var(--text)' }}>{s.t}</h3>
                  <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{s.d}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── INDUSTRY PORTAL CALLOUT ── */}
        <section className="py-16 px-4" style={{ background: 'var(--surface)' }}>
          <div className="max-w-4xl mx-auto">
            <div className="p-8 rounded-2xl flex flex-col md:flex-row gap-8 items-center" style={{
              background: 'linear-gradient(135deg, rgba(56,182,232,0.08), rgba(201,162,39,0.06))',
              border: '1px solid var(--border)',
            }}>
              <div className="flex-1">
                <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'var(--gold)' }}>Industry Portal</p>
                <h3 className="text-2xl md:text-3xl font-bold mb-3" style={{ color: 'var(--text)' }}>
                  Scouts, Labels & Promoters — built in.
                </h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                  Discover emerging talent, close deals, and manage artists directly on Vuka.
                  Built for recruiters, sync buyers, sponsors, and artist managers.
                </p>
              </div>
              <div className="flex-shrink-0 flex flex-col gap-3">
                <Link href="/industry" className="btn text-sm font-semibold px-6 py-3" style={{ background: 'var(--sky)', color: 'white', borderRadius: 12 }}>
                  Industry Portal <ArrowRight size={16} />
                </Link>
                <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>Free to apply</p>
              </div>
            </div>
          </div>
        </section>

        {/* ── FOR FANS ── */}
        <section className="py-24 px-4">
          <div className="max-w-6xl mx-auto">
            <div className="grid md:grid-cols-2 gap-16 items-center">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--sky)' }}>For Fans & Listeners</p>
                <h2 className="text-3xl md:text-5xl font-bold mb-6" style={{ color: 'var(--text)' }}>
                  Support the artists you love
                </h2>
                <p className="leading-relaxed mb-6" style={{ color: 'var(--text-muted)' }}>
                  Discover independent artists from across Africa and the diaspora. Buy their music directly — money goes straight to the artist.
                </p>
                <div className="space-y-3 mb-8">
                  {[
                    'Browse and stream previews before you buy',
                    'Secure checkout via Paystack or card',
                    'Instant download links in your inbox',
                    'Follow artists and get notified of new drops',
                    'Tip artists and back their recording goals',
                  ].map(item => (
                    <div key={item} className="flex items-center gap-3 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <div className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'rgba(42,157,92,0.12)' }}>
                        <span style={{ color: 'var(--green)', fontSize: 10 }}>✓</span>
                      </div>
                      {item}
                    </div>
                  ))}
                </div>
                <Link href="/auth/register?role=fan" className="btn btn-secondary inline-flex">
                  Create a Fan Account <ArrowRight size={16} />
                </Link>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { icon: Globe, title: 'Global Discovery', desc: 'Artists from SA, Nigeria, Ghana, Kenya and beyond.' },
                  { icon: TrendingUp, title: 'Support Goals', desc: "Back an artist's recording or tour fund directly." },
                  { icon: Music, title: 'Build a Library', desc: 'Your purchases available for download anytime.' },
                  { icon: Shield, title: 'Safe & Secure', desc: 'Protected checkout. Immediate delivery.' },
                ].map(c => (
                  <div key={c.title} className="p-5 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                    <c.icon size={20} className="mb-3" style={{ color: 'var(--sky)' }} />
                    <h4 className="font-semibold text-sm mb-1.5" style={{ color: 'var(--text)' }}>{c.title}</h4>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{c.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── PRICING ── */}
        <section className="py-24 px-4" style={{ background: 'var(--surface)' }}>
          <div className="max-w-5xl mx-auto text-center">
            <h2 className="text-3xl md:text-5xl font-bold mb-4" style={{ color: 'var(--text)' }}>
              Honest pricing. Always.
            </h2>
            <p className="mb-12" style={{ color: 'var(--text-muted)' }}>
              Start free. Upgrade when you're ready to keep more of what you earn.
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-10">
              {/* FREE */}
              <div className="p-8 rounded-2xl text-left flex flex-col" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <div className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>Free</div>
                <div className="text-5xl font-bold mb-1" style={{ color: 'var(--text)' }}>R0</div>
                <div className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>forever</div>
                <div className="text-2xl font-bold mb-1" style={{ color: 'var(--sky)' }}>85%</div>
                <div className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>artist keeps per sale</div>
                <div className="space-y-2 flex-1 mb-8">
                  {['Up to 2 releases/month', 'Beat store & licensing', 'Fan memberships', 'PDF license generation', 'Paystack + Flutterwave'].map(f => (
                    <div key={f} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <span style={{ color: 'var(--green)' }}>✓</span> {f}
                    </div>
                  ))}
                </div>
                <Link href="/auth/register" className="btn btn-secondary text-sm text-center w-full py-3">
                  Get Started Free
                </Link>
              </div>

              {/* PRO */}
              <div className="p-8 rounded-2xl text-left flex flex-col relative" style={{ background: 'var(--bg)', border: '2px solid var(--sky)', boxShadow: '0 0 30px rgba(56,182,232,0.1)' }}>
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-bold" style={{ background: 'var(--sky)', color: 'white' }}>Most Popular</div>
                <div className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--sky)' }}>Pro</div>
                <div className="text-5xl font-bold mb-1" style={{ color: 'var(--text)' }}>R249</div>
                <div className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>per month</div>
                <div className="text-2xl font-bold mb-1" style={{ color: 'var(--sky)' }}>92%</div>
                <div className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>artist keeps per sale</div>
                <div className="space-y-2 flex-1 mb-8">
                  {['Unlimited releases', '8% platform fee', 'Priority support', 'Advanced analytics', 'Industry marketplace access', 'Everything in Free'].map(f => (
                    <div key={f} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <span style={{ color: 'var(--green)' }}>✓</span> {f}
                    </div>
                  ))}
                </div>
                <Link href="/auth/register" className="btn btn-primary text-sm text-center w-full py-3">
                  Start Pro
                </Link>
              </div>

              {/* LABEL */}
              <div className="p-8 rounded-2xl text-left flex flex-col" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <div className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--gold)' }}>Label</div>
                <div className="text-5xl font-bold mb-1" style={{ color: 'var(--text)' }}>R999</div>
                <div className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>per month</div>
                <div className="text-2xl font-bold mb-1" style={{ color: 'var(--gold)' }}>95%</div>
                <div className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>artist keeps per sale</div>
                <div className="space-y-2 flex-1 mb-8">
                  {['Unlimited releases', '5% platform fee', 'Multiple artists under one account', 'Bulk payout management', 'White-label storefront', 'Everything in Pro'].map(f => (
                    <div key={f} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                      <span style={{ color: 'var(--green)' }}>✓</span> {f}
                    </div>
                  ))}
                </div>
                <Link href="/auth/register" className="btn btn-secondary text-sm text-center w-full py-3">
                  Start Label
                </Link>
              </div>
            </div>

            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              All plans include Paystack + Flutterwave payments, PDF license generation, secure download delivery, and full analytics. No hidden charges.
            </p>
          </div>
        </section>

        {/* ── FOOTER ── */}
        <footer className="py-10 px-4" style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
          <div className="max-w-6xl mx-auto flex flex-col items-center gap-6 md:flex-row md:justify-between">
            <div className="flex items-center gap-3">
              <span className="text-xl font-bold" style={{ color: 'var(--text)' }}>Vuka</span>
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>African music. Your money. Your terms.</span>
            </div>
            <div className="flex items-center gap-6 text-sm flex-wrap justify-center" style={{ color: 'var(--text-muted)' }}>
              <Link href="/store" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Store</Link>
              <Link href="/industry" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Industry</Link>
              <Link href="/legal/terms" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Terms</Link>
              <Link href="/legal/privacy" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Privacy</Link>
              <Link href="/legal/dmca" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>DMCA</Link>
              <Link href="/auth/login" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Log In</Link>
              <Link href="/auth/register" className="transition-colors hover:text-[var(--sky)]" style={{ color: 'var(--text-muted)' }}>Sign Up</Link>
            </div>
            <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
              © 2025 Vuka · Made in South Africa
            </p>
          </div>
        </footer>

      </main>
    </>
  );
}
