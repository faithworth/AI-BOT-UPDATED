import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import { createClient } from '@/lib/supabase';

export async function GET(req: NextRequest) {
  try {
    const supabase = createClient();
    const { data: { user }, error: authErr } = await supabase.auth.getUser();
    if (authErr || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const artist = await prisma.artist.findUnique({ where: { userId: user.id } });
    if (!artist) {
      return NextResponse.json({ payouts: [], pending: 0, processed: 0, failed: 0, methods: [] });
    }

    // ArtistPayout uses `amount` — not netAmount/fee (those don't exist on this model)
    const allPayouts = await prisma.artistPayout.findMany({
      where: { artistId: artist.id },
      orderBy: { createdAt: 'desc' },
      take: 100,
    });

    const pending   = allPayouts.filter(p => p.status === 'pending').reduce((s, p) => s + p.amount, 0);
    const processed = allPayouts.filter(p => p.status === 'paid').reduce((s, p) => s + p.amount, 0);
    const failed    = allPayouts.filter(p => p.status === 'failed').reduce((s, p) => s + p.amount, 0);

    // PayoutRequests (formal requests against the bank account system)
    const payoutRequests = await prisma.payoutRequest.findMany({
      where: { artistId: artist.id },
      orderBy: { createdAt: 'desc' },
      take: 50,
      include: { bankAccount: { select: { bankName: true, maskedNumber: true } } },
    });

    const methods: { type: string; name: string; configured: boolean }[] = [];
    if (artist.paystackRecipient) {
      methods.push({ type: 'paystack', name: 'Paystack (Direct to Bank)', configured: true });
    }
    // Stripe Connect removed (Phase 12) — Paystack only
    if (false) {
    }
    if (methods.length === 0) {
      methods.push({ type: 'paystack', name: 'Paystack (Direct to Bank)', configured: false });
    }

    return NextResponse.json({
      payouts: allPayouts.map(p => ({
        id: p.id,
        amount: p.amount,
        method: p.method,
        status: p.status,
        reference: p.reference,
        notes: p.notes,
        createdAt: p.createdAt,
        processedAt: p.processedAt,
      })),
      payoutRequests,
      balances: { pending, processed, failed, total: pending + processed },
      methods,
      artistHasPaystackRecipient: !!artist.paystackRecipient,
      artistHasStripeConnect: false, // Stripe removed
    });
  } catch (err) {
    console.error('Payouts fetch error:', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
