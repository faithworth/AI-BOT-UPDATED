'use client';
import { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { CheckCircle } from 'lucide-react';
import { Suspense } from 'react';

function ConnectReturnContent() {
  const router = useRouter();
  const params = useSearchParams();

  useEffect(() => {
    // If account param, we could verify, but just redirect to payouts
    const timer = setTimeout(() => router.push('/dashboard/payouts'), 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg)' }}>
      <div className="text-center p-8 rounded-2xl border max-w-md w-full" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <CheckCircle className="w-16 h-16 mx-auto mb-4" style={{ color: 'var(--green)' }} />
        <h1 className="text-2xl font-black mb-2" style={{ color: 'var(--text)' }}>Sharp! You're connected 🎉</h1>
        <p className="mb-4" style={{ color: 'var(--text-muted)' }}>Paystack is configured. You'll receive payouts directly to your bank.</p>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Redirecting to payouts…</p>
      </div>
    </div>
  );
}

export default function ConnectReturnPage() {
  return <Suspense><ConnectReturnContent /></Suspense>;
}
