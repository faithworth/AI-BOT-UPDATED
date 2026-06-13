/**
 * POST /api/support/webhook
 * Paystack webhook — confirms support/tip transactions.
  * Replaces the legacy /api/support/payfast-notify route.
 */

export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import { verifyPaystackWebhook, verifyTransaction } from '@/lib/paystack';
import { sendSupportFanConfirmation, sendSupportArtistNotification } from '@/lib/emails';
import { platformFee as calcFee, artistNet as calcNet } from '@/lib/plans';
import { logger } from '@/lib/logger';

export async function POST(req: NextRequest) {
  const traceId   = req.headers.get('x-trace-id') ?? 'no-trace';
  const signature = req.headers.get('x-paystack-signature') ?? '';
  const rawBody   = await req.text();

  if (!verifyPaystackWebhook(rawBody, signature)) {
    logger.warn('[support/webhook] Invalid signature', { traceId });
    return new NextResponse('Invalid signature', { status: 400 });
  }

  let event: any;
  try { event = JSON.parse(rawBody); } catch { return new NextResponse('Bad JSON', { status: 400 }); }

  if (event.event !== 'charge.success') return NextResponse.json({ ok: true });

  const reference = event.data?.reference ?? '';
  if (!reference.startsWith('SUP_')) return NextResponse.json({ ok: true });

  // Find pending txn by stored reference
  const txn = await prisma.supportTxn.findFirst({
    where: { paystackReference: reference },
    include: {
      artist: {
        include: {
          user:  true,
          goals: { where: { isActive: true }, take: 1 },
        },
      },
    },
  });

  if (!txn) {
    logger.warn('[support/webhook] SupportTxn not found', { traceId, reference });
    return NextResponse.json({ ok: true });
  }

  if (txn.status !== 'pending') {
    logger.info('[support/webhook] Duplicate — already processed', { traceId, reference });
    return NextResponse.json({ ok: true });
  }

  // Verify with Paystack
  try {
    const verification = await verifyTransaction(reference);
    if (verification.status !== 'success') return NextResponse.json({ ok: true });
  } catch (err) {
    logger.error('[support/webhook] Verification failed', { traceId, error: String(err) });
    return NextResponse.json({ ok: true });
  }

  try {
    await prisma.supportTxn.update({
      where: { id: txn.id },
      data:  { status: 'confirmed' },
    });

    const tipFee = calcFee(txn.amount, txn.artist.planSlug, txn.artist.planExpiresAt);
    const tipNet = calcNet(txn.amount, txn.artist.planSlug, txn.artist.planExpiresAt);

    await prisma.artistPayout.create({
      data: {
        artistId:  txn.artistId,
        amount:    tipNet,
        method:    'paystack',
        currency:  txn.currency,
        status:    'pending',
        reference,
        notes:     `Fan tip from ${txn.fanName} (fee: R${tipFee.toFixed(2)} kept by Vuka)`,
      },
    });

    const activeGoal = txn.artist.goals[0];
    if (activeGoal) {
      await prisma.goal.update({
        where: { id: activeGoal.id },
        data:  { currentAmount: { increment: txn.amount } },
      });
    }

    const goalPercent = activeGoal
      ? ((activeGoal.currentAmount + txn.amount) / activeGoal.targetAmount) * 100
      : undefined;

    await Promise.all([
      sendSupportFanConfirmation({ to: txn.fanEmail, fanName: txn.fanName, artistName: txn.artist.name, amount: txn.amount, currency: txn.currency, tier: txn.tier, message: txn.message || undefined }),
      sendSupportArtistNotification({ to: txn.artist.user.email, artistName: txn.artist.name, fanName: txn.fanName, amount: txn.amount, currency: txn.currency, tier: txn.tier, message: txn.message || undefined, goalTitle: activeGoal?.title, goalPercent }),
    ]);

    logger.info('[support/webhook] Tip confirmed', { traceId, txnId: txn.id });
  } catch (err) {
    logger.error('[support/webhook] Processing error', { traceId, error: String(err) });
  }

  return NextResponse.json({ ok: true });
}
