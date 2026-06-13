// ============================================================
// src/app/api/payouts/request/route.ts (Phase 9)
// Artist requests a payout — now sends sendPayoutRequested email
// ============================================================

export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireArtist } from '@/lib/auth';
import { requestPayout, retryPayoutRequest } from '@/lib/payouts';
import prisma from '@/lib/prisma';
import { schemas, validationError } from '@/lib/validation';
import { rateLimit, RATE_LIMITS, getClientIp } from '@/lib/rateLimit';
import { sendPayoutRequested } from '@/lib/emails';

const APP_URL = () => process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';

// GET — list artist's payout requests
export async function GET() {
  try {
    const user = await requireArtist();
    if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const requests = await prisma.payoutRequest.findMany({
      where: { artistId: user.artist.id },
      include: {
        bankAccount: {
          select: {
            bankName: true,
            accountHolder: true,
            maskedNumber: true,
            branchCode: true,
            accountType: true,
          },
        },
      },
      orderBy: { createdAt: 'desc' },
      take: 50,
    });

    return NextResponse.json({ requests });
  } catch (err) {
    console.error('[payouts/request] GET error:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 503 });
  }
}

// POST — submit a new payout request
export async function POST(req: NextRequest) {
  try {
    const user = await requireArtist();
    if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const ip = getClientIp(req.headers);
    const limited = await rateLimit(user.id, RATE_LIMITS.payout_request, ip);
    if (limited) return NextResponse.json({ error: 'Too many payout requests. Please wait before trying again.' }, { status: 429 });

    const raw = await req.json();
    const parsed = schemas.payout.request.safeParse(raw);
    if (!parsed.success) return validationError(parsed.error);

    const { amount, currency, method, bankAccountId, paypalEmail } = parsed.data;

    const request = await requestPayout({
      artistId: user.artist.id,
      amount,
      currency,
      method,
      bankAccountId,
      paypalEmail,
    });

    // Phase 9: notify artist payout request received
    try {
      await sendPayoutRequested({
        to: user.email,
        artistName: user.artist.name || user.name || 'Artist',
        amount: Number(amount),
        currency: currency || 'ZAR',
        payoutMethod: method === 'bank_transfer' ? 'Bank Transfer' : method === 'paypal' ? 'PayPal' : 'Paystack',
        referenceNumber: request.id,
        payoutsUrl: `${APP_URL()}/dashboard/payouts`,
      });
    } catch (e) { console.error('[payouts/request] email failed:', e); }

    return NextResponse.json({ request }, { status: 201 });
  } catch (err: any) {
    console.error('[payouts/request] POST error:', err?.message);
    const code = err?.message?.includes('exceeds available') ? 422 : 503;
    return NextResponse.json({ error: err?.message || 'Payout request failed' }, { status: code });
  }
}

// PATCH — retry a failed payout request
export async function PATCH(req: NextRequest) {
  try {
    const user = await requireArtist();
    if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const { requestId } = await req.json();
    if (!requestId) return NextResponse.json({ error: 'requestId required' }, { status: 400 });

    // Verify ownership
    const existing = await prisma.payoutRequest.findFirst({
      where: { id: requestId, artistId: user.artist.id },
    });
    if (!existing) return NextResponse.json({ error: 'Payout request not found' }, { status: 404 });

    const result = await retryPayoutRequest(requestId);
    return NextResponse.json({ result });
  } catch (err: any) {
    console.error('[payouts/request] PATCH error:', err?.message);
    const code = err?.message?.includes('Max retry') ? 409 : 503;
    return NextResponse.json({ error: err?.message || 'Retry failed' }, { status: code });
  }
}
