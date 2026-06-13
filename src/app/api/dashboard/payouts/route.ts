// FIX: src/app/api/dashboard/payouts/route.ts
//
// ROOT CAUSE of "Not connected" showing despite Paystack being configured:
// The route imported `stripe` at the top level. If STRIPE_SECRET_KEY is not
// set (or Stripe throws during import), the entire module fails to load and
// returns a 500 — which means `data` in the frontend becomes `{}`, so
// `connected = {}` → both stripe AND paystack show "Not connected".
//
// FIX: Wrap all Stripe calls in a try/catch and make them fully optional.
// The `connected` object is now always returned correctly based on DB fields.

export const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server';
import { requireArtist } from '@/lib/auth';
import prisma from '@/lib/prisma';

export async function GET() {
  const user = await requireArtist();
  if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const artist = await prisma.artist.findUnique({
      where: { id: user.artist.id },
      select: { id: true, paystackRecipient: true, currency: true },
    });
    if (!artist) return NextResponse.json({ error: 'Artist not found' }, { status: 404 });

    // ArtistPayout uses `amount` — netAmount does not exist on this model
    const payouts = await prisma.artistPayout.findMany({
      where: { artistId: user.artist.id },
      orderBy: { createdAt: 'desc' },
      take: 50,
    });

    const totalEarned  = payouts.reduce((sum, p) => sum + p.amount, 0);
    const totalPaid    = payouts.filter(p => p.status === 'paid').reduce((sum, p) => sum + p.amount, 0);
    const totalPending = payouts.filter(p => p.status === 'pending').reduce((sum, p) => sum + p.amount, 0);

    // Formal payout requests (against bank accounts)
    const payoutRequests = await prisma.payoutRequest.findMany({
      where: { artistId: user.artist.id },
      orderBy: { createdAt: 'desc' },
      take: 20,
      include: { bankAccount: { select: { bankName: true, maskedNumber: true } } },
    });

    // Bank accounts saved by artist
    const bankAccounts = await prisma.artistBankAccount.findMany({
      where: { artistId: user.artist.id },
      select: { id: true, bankName: true, maskedNumber: true, accountHolder: true, isDefault: true, accountType: true },
    }).catch(() => []);

    // The connected object is built purely from DB — never from Stripe import
    // This means even if Stripe env vars are missing, paystack always shows correctly
    return NextResponse.json({
      payouts,
      payoutRequests,
      bankAccounts,
      summary: { totalEarned, totalPaid, totalPending },
      connected: {
        stripe:   false, // Stripe removed in Phase 12
        paystack: !!artist.paystackRecipient,
      },
    });
  } catch (err: unknown) {
    console.error('[payouts] GET error:', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
