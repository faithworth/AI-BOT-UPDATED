/**
 * POST /api/checkout/paystack/webhook
 *
 * THE single Paystack webhook URL — register ONLY this URL in the Paystack
 * Dashboard (Settings → API Keys & Webhooks). Paystack supports exactly one
 * webhook URL per account, so this route dispatches by reference prefix:
 *
 *   VKB_  → beat/release/video/sample/merch purchase (handled below)
 *   PLAN_ → artist plan subscription   → handlePlanEvent
 *   MKT_  → marketplace service order  → handleMarketplaceEvent
 *   MEM_  → fan creator membership     → handleMembershipEvent
 *   ISO_  → industry service order     → handleIndustryOrderEvent
 *
 * Replaces /api/checkout/payfast/notify.
 * Handles beat, release, video, sample, merch purchase confirmation.
 * Reference stored in purchase.paystackReference during initialize.
 */

export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import { verifyPaystackWebhook, verifyTransaction } from '@/lib/paystack';
import { generateLicensePDF } from '@/lib/pdf';
import { uploadBuffer, r2Keys, getPublicUrl } from '@/lib/r2';
import { sendPurchaseConfirmation, sendArtistSaleNotification } from '@/lib/emails';
import { auditLog } from '@/lib/audit';
import { logger } from '@/lib/logger';
import { incrementDailyRollup } from '@/lib/social';
import { platformFee as calcPlatformFee } from '@/lib/plans';
import { handlePlanEvent, handleMarketplaceEvent, handleMembershipEvent, handleIndustryOrderEvent } from '@/lib/webhooks/paystack-handlers';

export async function POST(req: NextRequest) {
  const traceId   = req.headers.get('x-trace-id') ?? 'no-trace';
  const signature = req.headers.get('x-paystack-signature') ?? '';
  const rawBody   = await req.text();

  if (!verifyPaystackWebhook(rawBody, signature)) {
    logger.warn('[paystack/webhook] Invalid signature', { traceId });
    return new NextResponse('Invalid signature', { status: 400 });
  }

  let event: any;
  try { event = JSON.parse(rawBody); } catch { return new NextResponse('Bad JSON', { status: 400 }); }

  if (event.event !== 'charge.success') return NextResponse.json({ ok: true });

  const reference = event.data?.reference ?? '';

  // ── Single-webhook dispatch ────────────────────────────────────────────
  // Paystack only supports ONE webhook URL per account. This route is that
  // URL — dispatch to the right handler based on the reference prefix set
  // by generateReference() in each checkout flow.
  if (reference.startsWith('PLAN_')) {
    await handlePlanEvent(event, traceId);
    return NextResponse.json({ ok: true });
  }
  if (reference.startsWith('MKT_')) {
    await handleMarketplaceEvent(event, traceId);
    return NextResponse.json({ ok: true });
  }
  if (reference.startsWith('MEM_')) {
    await handleMembershipEvent(event, traceId);
    return NextResponse.json({ ok: true });
  }
  if (reference.startsWith('ISO_')) {
    await handleIndustryOrderEvent(event, traceId);
    return NextResponse.json({ ok: true });
  }

  // Find purchase by stored reference
  const purchase = await prisma.purchase.findFirst({
    where: { paystackReference: reference },
  });

  if (!purchase) {
    logger.warn('[paystack/webhook] Purchase not found for reference', { traceId, reference });
    return NextResponse.json({ ok: true });
  }

  if (purchase.status !== 'pending') {
    logger.info('[paystack/webhook] Duplicate — already processed', { traceId, reference });
    return NextResponse.json({ ok: true });
  }

  // Verify via Paystack API (don't trust webhook payload amount alone)
  let verification;
  try {
    verification = await verifyTransaction(reference);
  } catch (err) {
    logger.error('[paystack/webhook] Verification failed', { traceId, reference, error: String(err) });
    return NextResponse.json({ ok: true });
  }

  if (verification.status !== 'success') return NextResponse.json({ ok: true });

  // Amount check
  if (Math.abs(verification.amountZAR - purchase.amount) > 0.01) {
    logger.error('[paystack/webhook] Amount mismatch', { traceId, paid: verification.amountZAR, expected: purchase.amount });
    await auditLog.securityEvent('security.invalid_download_attempt', `Amount mismatch purchaseId=${purchase.id}`, 'paystack');
    return new NextResponse('Amount mismatch', { status: 400 });
  }

  // Resolve plan for correct fee
  let artistPlanSlug: string | null = null;
  let artistPlanExpiresAt: Date | null = null;
  if (purchase.beatId) {
    const beat = await prisma.beat.findUnique({ where: { id: purchase.beatId }, select: { artist: { select: { planSlug: true, planExpiresAt: true } } } });
    artistPlanSlug = beat?.artist?.planSlug ?? null;
    artistPlanExpiresAt = beat?.artist?.planExpiresAt ?? null;
  } else if ((purchase as any).releaseId) {
    const rel = await prisma.release.findUnique({ where: { id: (purchase as any).releaseId }, select: { artist: { select: { planSlug: true, planExpiresAt: true } } } });
    artistPlanSlug = rel?.artist?.planSlug ?? null;
    artistPlanExpiresAt = rel?.artist?.planExpiresAt ?? null;
  }

  const platformFeeAmt = calcPlatformFee(purchase.amount, artistPlanSlug, artistPlanExpiresAt);
  const netAmount      = Math.round((purchase.amount - platformFeeAmt) * 100) / 100;

  // Link userId if not set
  let resolvedUserId = purchase.userId;
  if (!resolvedUserId && purchase.buyerEmail) {
    const buyer = await prisma.user.findUnique({ where: { email: purchase.buyerEmail }, select: { id: true } }).catch(() => null);
    resolvedUserId = buyer?.id ?? null;
  }

  await prisma.purchase.update({
    where: { id: purchase.id },
    data: {
      status: 'confirmed',
      platformFee: platformFeeAmt,
      netAmount,
      ...(resolvedUserId && !purchase.userId ? { userId: resolvedUserId } : {}),
    },
  });

  const appUrl      = process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';
  const downloadUrl = `${appUrl}/download/${purchase.downloadToken}`;

  let itemName    = 'your purchase';
  let artistEmail = '';
  let artistName  = '';
  let artworkUrl  = '';
  let artistId    = '';
  let licenseUrl  = '';

  if (purchase.itemType === 'beat' && purchase.beatId) {
    const beat = await prisma.beat.findUnique({ where: { id: purchase.beatId }, include: { artist: { include: { user: true } } } });
    if (beat) {
      itemName = beat.title; artistEmail = beat.artist.user.email;
      artistName = beat.artist.name; artworkUrl = beat.artworkUrl || ''; artistId = beat.artist.id;
      try {
        const pdfBuffer = await generateLicensePDF({ licenseId: purchase.licenseId, licenseType: purchase.licenseType, beatTitle: beat.title, artistName: beat.artist.name, buyerName: purchase.buyerName, buyerEmail: purchase.buyerEmail, amount: purchase.amount, currency: purchase.currency, date: new Date() });
        const pdfKey = r2Keys.license(purchase.licenseId);
        await uploadBuffer(pdfKey, pdfBuffer, 'application/pdf');
        licenseUrl = getPublicUrl(pdfKey);
        await prisma.purchase.update({ where: { id: purchase.id }, data: { licenseUrl } });
      } catch (e) { logger.error('[paystack/webhook] PDF failed', { traceId, error: String(e) }); }
      if (purchase.licenseType === 'exclusive') {
        await prisma.beat.update({ where: { id: beat.id }, data: { isExclusive: true, isActive: false } });
        await auditLog.exclusiveLocked(beat.id, beat.title, purchase.id);
      }
      await prisma.beat.update({ where: { id: beat.id }, data: { sales: { increment: 1 } } });
      await incrementDailyRollup(artistId, 'beatSales').catch(() => {});
      await incrementDailyRollup(artistId, 'revenue').catch(() => {});
    }
  } else if (purchase.itemType === 'release' && (purchase as any).releaseId) {
    const release = await prisma.release.findUnique({ where: { id: (purchase as any).releaseId }, include: { artist: { include: { user: true } } } });
    if (release) {
      itemName = release.title; artistEmail = release.artist.user.email;
      artistName = release.artist.name; artworkUrl = release.artworkUrl || ''; artistId = release.artist.id;
      await prisma.release.update({ where: { id: release.id }, data: { sales: { increment: 1 } } });
      await incrementDailyRollup(artistId, 'releaseSales').catch(() => {});
      await incrementDailyRollup(artistId, 'revenue').catch(() => {});
    }
  } else if (purchase.itemType === 'video' && purchase.videoId) {
    const video = await prisma.video.findUnique({ where: { id: purchase.videoId }, include: { artist: { include: { user: true } } } });
    if (video) {
      itemName = video.title; artistEmail = video.artist.user.email;
      artistName = video.artist.name; artworkUrl = video.thumbnailUrl || ''; artistId = video.artist.id;
      await prisma.video.update({ where: { id: video.id }, data: { sales: { increment: 1 } } });
      await incrementDailyRollup(artistId, 'revenue').catch(() => {});
    }
  } else if (purchase.itemType === 'sample' && purchase.sampleId) {
    const sample = await prisma.sample.findUnique({ where: { id: purchase.sampleId }, include: { artist: { include: { user: true } } } });
    if (sample) {
      itemName = sample.title; artistEmail = sample.artist.user.email;
      artistName = sample.artist.name; artworkUrl = sample.artworkUrl || ''; artistId = sample.artist.id;
      await prisma.sample.update({ where: { id: sample.id }, data: { sales: { increment: 1 } } });
      await incrementDailyRollup(artistId, 'revenue').catch(() => {});
    }
  } else if (purchase.itemType === 'merch' && (purchase as any).merchId) {
    const merch = await prisma.merch.findUnique({ where: { id: (purchase as any).merchId }, include: { artist: { include: { user: true } } } });
    if (merch) {
      itemName = merch.title; artistEmail = merch.artist.user.email;
      artistName = merch.artist.name; artworkUrl = merch.imageUrl || ''; artistId = merch.artist.id;
      await prisma.merch.update({ where: { id: merch.id }, data: { stock: { decrement: 1 } } }).catch(() => {});
      await incrementDailyRollup(artistId, 'revenue').catch(() => {});
    }
  }

  if (artistId) {
    await prisma.artistPayout.create({
      data: { artistId, purchaseId: purchase.id, amount: netAmount, method: 'paystack', currency: purchase.currency, status: 'pending', reference, notes: `${purchase.itemType} sale via Paystack — ${itemName}` },
    });
  }

  await auditLog.purchaseConfirmed(purchase.id, itemName, purchase.amount, purchase.currency, purchase.buyerEmail);

  const freshPurchase = await prisma.purchase.findUnique({ where: { id: purchase.id }, select: { receiptUrl: true } });
  if (freshPurchase?.receiptUrl !== 'email:sent') {
    try {
      await sendPurchaseConfirmation({ to: purchase.buyerEmail, buyerName: purchase.buyerName, itemName, itemType: purchase.itemType, licenseType: purchase.licenseType || undefined, downloadUrl, amount: purchase.amount, currency: purchase.currency, licenseId: purchase.licenseId, artworkUrl: artworkUrl || undefined, licenseUrl: licenseUrl || undefined });
      await prisma.purchase.update({ where: { id: purchase.id }, data: { receiptUrl: 'email:sent' } });
    } catch (e) { logger.error('[paystack/webhook] Buyer email failed', { traceId, error: String(e) }); }
  }

  if (artistEmail) {
    try {
      await sendArtistSaleNotification({ to: artistEmail, artistName, buyerName: purchase.buyerName, itemName, licenseType: purchase.licenseType || undefined, amount: purchase.amount, currency: purchase.currency, dashboardUrl: `${appUrl}/dashboard` });
    } catch (e) { logger.error('[paystack/webhook] Artist email failed', { traceId, error: String(e) }); }
  }

  logger.info('[paystack/webhook] Purchase confirmed', { traceId, purchaseId: purchase.id });
  return NextResponse.json({ ok: true });
}
