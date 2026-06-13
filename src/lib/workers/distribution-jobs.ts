/**
 * VUKA — Distribution Queue Worker (Phase 6)
 *
 * processDistributionQueue:
 *   - Finds DSPDelivery rows with status 'queued' or timed-out 'submitting'
 *   - Retries failed deliveries (up to MAX_RETRIES)
 *   - Advances release to 'live' when all DSPs report live/submitted
 *   - Sends artist notification on first live delivery
 *
 * autoDispatchApprovedPayouts:
 *   - Finds PayoutRequests with status 'approved'
 *   - Dispatches each via the payout processor (Paystack / Flutterwave / PayPal)
 *   - Runs Monday–Friday via cron
 */

import prisma from '../prisma';
import { logger } from '../logger';
import { deliverToDsp, DspDeliveryInput } from '../distribution';
import { dispatchPayout } from '../earnings';
import { createNotification } from '../social';

const MAX_RETRIES = 3;
const SUBMITTING_TIMEOUT_MS = 30 * 60 * 1000; // 30 min — re-attempt stuck submitting

// ── DISTRIBUTION QUEUE PROCESSOR ─────────────────────────────

export async function processDistributionQueue(): Promise<{
  processed: number;
  succeeded: number;
  failed: number;
  durationMs: number;
}> {
  const start = Date.now();
  let processed = 0;
  let succeeded = 0;
  let failed = 0;

  const timeoutThreshold = new Date(Date.now() - SUBMITTING_TIMEOUT_MS);

  // Find queued deliveries and timed-out submitting ones
  const pendingDeliveries = await prisma.dSPDelivery.findMany({
    where: {
      OR: [
        { status: 'queued' },
        { status: 'submitting', lastRetryAt: { lt: timeoutThreshold } },
        { status: 'failed', retryCount: { lt: MAX_RETRIES } },
      ],
    },
    include: {
      distributionRelease: {
        include: {
          tracks: true,
          artist: { select: { id: true, name: true, userId: true } },
        },
      },
    },
    orderBy: { createdAt: 'asc' },
    take: 50, // Process in batches to avoid timeout
  });

  for (const delivery of pendingDeliveries) {
    const release = delivery.distributionRelease;
    if (!release || !['approved', 'delivering', 'submitted'].includes(release.status)) {
      continue;
    }

    processed++;

    try {
      // Mark as submitting
      await prisma.dSPDelivery.update({
        where: { id: delivery.id },
        data: {
          status: 'submitting',
          lastRetryAt: new Date(),
          retryCount: { increment: delivery.status === 'failed' ? 1 : 0 },
        },
      });

      const artistName = release.artistName || release.artist?.name || '';
      const releaseDate = (release as Record<string, unknown>).scheduledDate as Date
        ?? (release as Record<string, unknown>).scheduledFor as Date
        ?? new Date();

      const adapterInput: DspDeliveryInput = {
        releaseId: release.id,
        artistName,
        title: release.title,
        releaseType: release.releaseType as 'single' | 'ep' | 'album',
        releaseDate,
        upc: release.upc || '',
        artworkUrl: release.artworkUrl,
        tracks: release.tracks.map((t) => ({
          trackNumber: t.trackNumber,
          title: t.title,
          isrc: t.isrc || '',
          audioUrl: t.masterFileUrl || '',
          durationSeconds: t.duration || 0,
        })),
      };

      const result = await deliverToDsp(delivery.dsp, adapterInput);

      const finalStatus =
        result.status === 'delivered' ? 'submitted' :
        result.status === 'queued' ? 'submitted' :
        result.status;

      await prisma.dSPDelivery.update({
        where: { id: delivery.id },
        data: {
          status: finalStatus,
          submittedAt: ['submitted', 'delivered'].includes(finalStatus) ? new Date() : undefined,
          errorMessage: result.error || '',
          ...(result.externalId ? { dspReferenceId: result.externalId } : {}),
        },
      });

      if (finalStatus === 'submitted') {
        succeeded++;
        logger.info('[dist-worker] DSP delivery succeeded', {
          releaseId: release.id,
          dsp: delivery.dsp,
        });
      } else {
        failed++;
      }
    } catch (err: unknown) {
      failed++;
      const msg = err instanceof Error ? err.message : String(err);

      await prisma.dSPDelivery.update({
        where: { id: delivery.id },
        data: {
          status: delivery.retryCount + 1 >= MAX_RETRIES ? 'failed' : 'queued',
          errorMessage: msg,
          failedAt: new Date(),
          retryCount: { increment: 1 },
        },
      });

      logger.error('[dist-worker] DSP delivery failed', {
        releaseId: release.id,
        dsp: delivery.dsp,
        error: msg,
        retryCount: delivery.retryCount + 1,
      });
    }
  }

  // After processing, check if any releases are now fully submitted
  await promoteSubmittedReleases();

  const durationMs = Date.now() - start;
  logger.info('[dist-worker] processDistributionQueue complete', {
    processed, succeeded, failed, durationMs,
  });

  return { processed, succeeded, failed, durationMs };
}

// After a delivery batch, advance release status and notify artists
async function promoteSubmittedReleases(): Promise<void> {
  // Find releases in delivering state where all DSPs are submitted or live
  const deliveringReleases = await prisma.distributionRelease.findMany({
    where: { status: 'delivering' },
    include: {
      dspDeliveries: true,
      artist: { select: { userId: true, name: true } },
    },
  });

  for (const release of deliveringReleases) {
    if (!release.dspDeliveries.length) continue;

    const allSubmitted = release.dspDeliveries.every((d) =>
      ['submitted', 'live', 'delivered'].includes(d.status)
    );

    if (allSubmitted) {
      await prisma.distributionRelease.update({
        where: { id: release.id },
        data: {
          status: 'submitted',
          submittedAt: new Date(),
        },
      });

      // Notify artist
      if (release.artist?.userId) {
        await createNotification({
          userId: release.artist.userId,
          type: 'release_submitted',
          title: '🎵 Your release is live!',
          body: `"${release.title}" has been submitted to ${release.dspDeliveries.length} platform${release.dspDeliveries.length === 1 ? '' : 's'}. It will go live within 24–72 hours.`,
          linkType: 'distribution_release',
          linkId: release.id,
        }).catch(() => null);
      }
    }
  }
}

// ── PAYOUT AUTO-DISPATCHER ────────────────────────────────────

export async function autoDispatchApprovedPayouts(): Promise<{
  dispatched: number;
  failed: number;
  durationMs: number;
}> {
  const start = Date.now();
  let dispatched = 0;
  let failed = 0;

  // Only dispatch if it's a weekday (Mon–Fri)
  const dayOfWeek = new Date().getDay(); // 0=Sun, 6=Sat
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    logger.info('[payout-worker] Skipping auto-dispatch — weekend');
    return { dispatched: 0, failed: 0, durationMs: Date.now() - start };
  }

  const approvedRequests = await prisma.payoutRequest.findMany({
    where: {
      status: 'approved',
      // Only dispatch requests approved more than 30 min ago (gives admin time to cancel)
      approvedAt: { lt: new Date(Date.now() - 30 * 60 * 1000) },
    },
    take: 20, // Cap per run to avoid overload
    orderBy: { approvedAt: 'asc' },
  });

  for (const request of approvedRequests) {
    try {
      const result = await dispatchPayout(request.id);

      if (result.success) {
        dispatched++;
        logger.info('[payout-worker] Payout dispatched', {
          requestId: request.id,
          amount: request.amount,
          currency: request.currency,
          referenceId: result.referenceId,
        });
      } else {
        failed++;
        logger.error('[payout-worker] Payout dispatch failed', {
          requestId: request.id,
          error: result.error,
        });
      }
    } catch (err: unknown) {
      failed++;
      const msg = err instanceof Error ? err.message : String(err);
      logger.error('[payout-worker] Payout exception', { requestId: request.id, error: msg });
    }
  }

  const durationMs = Date.now() - start;
  logger.info('[payout-worker] autoDispatchApprovedPayouts complete', {
    dispatched, failed, durationMs,
  });

  return { dispatched, failed, durationMs };
}
