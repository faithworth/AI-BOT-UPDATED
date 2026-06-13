import Link from 'next/link';
import { Music2 } from 'lucide-react';

export const metadata = { title: 'Refund Policy — Vuka', description: 'Vuka refund and return policy for digital goods.' };

export default function RefundsPage() {
  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <header className="px-6 py-4 flex items-center gap-3" style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <Link href="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'var(--sky)' }}><Music2 size={13} className="text-white" /></div>
          <span className="font-bold" style={{ color: 'var(--text)' }}>Vuka</span>
        </Link>
        <span style={{ color: 'var(--border)' }}>/</span>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Refund Policy</span>
      </header>
      <main className="max-w-3xl mx-auto px-6 py-12">
        <h1 className="text-3xl font-black mb-2" style={{ color: 'var(--text)', fontFamily: 'var(--font-display)' }}>Refund Policy</h1>
        <p className="text-sm mb-10" style={{ color: 'var(--text-muted)' }}>Last updated: January 2025</p>

        <div className="p-5 rounded-2xl mb-8" style={{ background: 'rgba(201,162,39,0.08)', border: '1px solid rgba(201,162,39,0.25)' }}>
          <p className="font-bold mb-1" style={{ color: 'var(--gold)' }}>All sales are final</p>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Vuka sells digital goods. Once a purchase is confirmed and download access is granted, the transaction is complete and non-refundable. This is consistent with how all major digital music platforms operate (Beatstars, TuneCore, Bandcamp).</p>
        </div>

        <div className="space-y-6 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          <div>
            <h2 className="text-base font-bold mb-2" style={{ color: 'var(--text)' }}>Why no refunds?</h2>
            <p>Digital files are delivered instantly upon payment confirmation. Unlike physical goods, digital files cannot be "returned" — once downloaded, the buyer retains a copy regardless. Under the South African Electronic Communications and Transactions Act (ECT Act), sellers may exclude the right of return for digital goods delivered immediately on purchase.</p>
          </div>

          <div>
            <h2 className="text-base font-bold mb-2" style={{ color: 'var(--text)' }}>Exceptions we will honour</h2>
            <p>We will issue a credit or re-grant access if: (a) you were charged but did not receive download access; (b) the file you downloaded is corrupted and we cannot provide a working replacement; (c) you were charged twice for the same item.</p>
            <p className="mt-2">To raise an exception, email <a href="mailto:support@vuka.co.za" className="underline" style={{ color: 'var(--sky)' }}>support@vuka.co.za</a> within 7 days with your order reference.</p>
          </div>

          <div>
            <h2 className="text-base font-bold mb-2" style={{ color: 'var(--text)' }}>Exclusive licences</h2>
            <p>Exclusive licence purchases are irreversible. Once an exclusive licence is confirmed, the beat is permanently locked and removed from sale. No refund will be issued for exclusive purchases under any circumstances.</p>
          </div>

          <div>
            <h2 className="text-base font-bold mb-2" style={{ color: 'var(--text)' }}>Chargebacks</h2>
            <p>Filing a fraudulent chargeback for a digital good you received and downloaded constitutes fraud. We will contest all fraudulent chargebacks and may report repeat offenders to Paystack and relevant authorities.</p>
          </div>
        </div>

        <div className="mt-10 pt-8" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            <Link href="/legal/terms" className="underline" style={{ color: 'var(--sky)' }}>Terms of Service</Link>
            {' · '}<Link href="/legal/privacy" className="underline" style={{ color: 'var(--sky)' }}>Privacy Policy</Link>
            {' · '}<Link href="/legal/dmca" className="underline" style={{ color: 'var(--sky)' }}>DMCA</Link>
          </p>
        </div>
      </main>
    </div>
  );
}
