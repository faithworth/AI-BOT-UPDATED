/**
 * POST /api/purchase/[id]/send-confirmation
 *
 * Called by the success page when it detects a purchase has become 'confirmed'.
  * Acts as a belt-and-suspenders fallback in case the Paystack webhook notify route
 * either timed out before reaching the email send, or the Resend call failed.
 *
 * Deduplication: uses `receiptUrl` field as a sent-sentinel.
 * If receiptUrl === 'email:sent', the confirmation was already emailed — skip.
 * Otherwise: send the email and mark receiptUrl = 'email:sent'.
 *
 * Safe to call multiple times — idempotent.
 */

import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import { sendPurchaseConfirmation } from '@/lib/emails';
import { logger } from '@/lib/logger';

export const dynamic = 'force-dynamic';

const EMAIL_SENT_SENTINEL = 'email:sent';

export async function POST(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params;
  if (!id) return NextResponse.json({ error: 'Missing id' }, { status: 400 });

  try {
    const purchase = await prisma.purchase.findUnique({
      where: { id },
      include: {
        beat:               { select: { title: true, artworkUrl: true } },
        release:            { select: { title: true, artworkUrl: true } },
        distributionRelease:{ select: { title: true, artworkUrl: true } },
        video:              { select: { title: true, thumbnailUrl: true } },
        sample:             { select: { title: true, artworkUrl: true } },
      },
    });

    if (!purchase) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    // Only send for confirmed purchases
    if (purchase.status !== 'confirmed') {
      return NextResponse.json({ skipped: 'not_confirmed' });
    }

    // Idempotency — already sent
    if (purchase.receiptUrl === EMAIL_SENT_SENTINEL) {
      return NextResponse.json({ skipped: 'already_sent' });
    }

    const appUrl     = process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';
    const downloadUrl = `${appUrl}/download/${purchase.downloadToken}`;

    const itemName =
      purchase.beat?.title ||
      purchase.release?.title ||
      (purchase as any).distributionRelease?.title ||
      purchase.video?.title ||
      purchase.sample?.title ||
      'your purchase';

    const artworkUrl =
      purchase.beat?.artworkUrl ||
      purchase.release?.artworkUrl ||
      (purchase as any).distributionRelease?.artworkUrl ||
      (purchase as any).sample?.artworkUrl ||
      (purchase as any).video?.thumbnailUrl ||
      undefined;

    await sendPurchaseConfirmation({
      to:          purchase.buyerEmail,
      buyerName:   purchase.buyerName,
      itemName,
      itemType:    purchase.itemType,
      licenseType: purchase.licenseType || undefined,
      downloadUrl,
      amount:      purchase.amount,
      currency:    purchase.currency,
      licenseId:   purchase.licenseId,
      artworkUrl:  artworkUrl || undefined,
      licenseUrl:  purchase.licenseUrl || undefined,
    });

    // Mark as sent so notify (if it runs later) doesn't double-send
    await prisma.purchase.update({
      where: { id },
      data:  { receiptUrl: EMAIL_SENT_SENTINEL },
    });

    logger.info('[send-confirmation] Fallback email sent', { purchaseId: id });
    return NextResponse.json({ ok: true });

  } catch (err) {
    logger.error('[send-confirmation] Failed', {
      purchaseId: id,
      error: err instanceof Error ? err.message : String(err),
    });
    return NextResponse.json({ error: 'Email failed' }, { status: 500 });
  }
}
