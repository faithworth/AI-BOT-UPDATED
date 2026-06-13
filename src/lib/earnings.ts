// src/lib/earnings.ts
// Phase 7 — Royalty & Earnings Processing
//
// Covers:
//   1. DSP report CSV parser + column mapper
//   2. Revenue share calculation (per subscription plan)
//   3. Earnings ingestion pipeline (validate → preview → confirm → credit)
//   4. PayFast / Flutterwave / PayPal payout processors
//   5. Webhook status handlers for all three providers

import prisma from './prisma';
import { logger } from './logger';
import { getPlan } from './plans';

// ── 1. DSP REPORT COLUMN MAPS ─────────────────────────────────
// Each DSP exports CSV with different column names.
// Map DSP slug → { isrc, streams, grossAmount, currency, period, territory }

export interface EarningsRow {
  isrc: string;
  streams: number;
  grossAmount: number;
  currency: string;
  period: string;       // YYYY-MM
  territory?: string;
  platformName: string;
}

type ColumnMap = {
  isrc: string;
  streams: string;
  grossAmount: string;
  currency?: string;
  period?: string;
  territory?: string;
};

const DSP_COLUMN_MAPS: Record<string, ColumnMap> = {
  spotify: {
    isrc:        'ISRC',
    streams:     'Quantity',
    grossAmount: 'Earnings (ZAR)',
    currency:    'Currency',
    period:      'Reporting Period',
    territory:   'Country',
  },
  apple_music: {
    isrc:        'ISRC',
    streams:     'Units',
    grossAmount: 'Royalty Price',
    currency:    'Royalty Currency',
    period:      'Start Date',
    territory:   'Country Of Sale',
  },
  youtube_music: {
    isrc:        'ISRC',
    streams:     'Views',
    grossAmount: 'Revenue (USD)',
    currency:    'Currency',
    period:      'Period',
    territory:   'Country',
  },
  audiomack: {
    isrc:        'ISRC',
    streams:     'Plays',
    grossAmount: 'Earnings',
    currency:    'Currency',
    period:      'Month',
    territory:   'Country',
  },
  boomplay: {
    isrc:        'ISRC',
    streams:     'Streams',
    grossAmount: 'Revenue',
    currency:    'Currency',
    period:      'Period',
    territory:   'Country',
  },
  deezer: {
    isrc:        'isrc',
    streams:     'streams',
    grossAmount: 'gross_revenue',
    currency:    'currency',
    period:      'reporting_month',
    territory:   'territory',
  },
  tidal: {
    isrc:        'ISRC',
    streams:     'Streams',
    grossAmount: 'Gross Revenue',
    currency:    'Currency',
    period:      'Period',
    territory:   'Territory',
  },
  // Generic fallback — used when DSP isn't in the map
  generic: {
    isrc:        'ISRC',
    streams:     'Streams',
    grossAmount: 'Gross Amount',
    currency:    'Currency',
    period:      'Period',
  },
};

// ── CSV Parser ────────────────────────────────────────────────

function parseCSV(raw: string): Record<string, string>[] {
  const lines = raw.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  if (lines.length < 2) return [];

  // Find header row — first non-empty line
  const headerLine = lines.find((l) => l.trim().length > 0) || lines[0];
  const headers = parseCSVLine(headerLine);
  const rows: Record<string, string>[] = [];

  for (let i = lines.indexOf(headerLine) + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = parseCSVLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h.trim()] = (values[idx] || '').trim();
    });
    rows.push(row);
  }
  return rows;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current);
  return result;
}

function normalisePeriod(raw: string): string {
  // Accepts: "2024-03", "March 2024", "2024/03", "2024-03-01"
  if (!raw) return new Date().toISOString().slice(0, 7);
  const cleaned = raw.trim();
  if (/^\d{4}-\d{2}$/.test(cleaned)) return cleaned;
  if (/^\d{4}\/\d{2}$/.test(cleaned)) return cleaned.replace('/', '-');
  if (/^\d{4}-\d{2}-\d{2}/.test(cleaned)) return cleaned.slice(0, 7);
  // "March 2024" → "2024-03"
  const months: Record<string, string> = {
    january: '01', february: '02', march: '03', april: '04',
    may: '05', june: '06', july: '07', august: '08',
    september: '09', october: '10', november: '11', december: '12',
    jan: '01', feb: '02', mar: '03', apr: '04',
    jun: '06', jul: '07', aug: '08', sep: '09', oct: '10', nov: '11', dec: '12',
  };
  const parts = cleaned.toLowerCase().split(' ');
  if (parts.length === 2) {
    const [a, b] = parts;
    if (months[a] && /^\d{4}$/.test(b)) return `${b}-${months[a]}`;
    if (months[b] && /^\d{4}$/.test(a)) return `${a}-${months[b]}`;
  }
  return cleaned.slice(0, 7);
}

// ── 2. REVENUE SHARE CALCULATION ─────────────────────────────
// Fetches the artist's active subscription plan and calculates
// Vuka's fee + artist net amount.

export async function calculateRevenueShare(params: {
  artistId: string;
  grossAmount: number;
  currency: string;
}): Promise<{
  grossAmount: number;
  vukaFeePercent: number;
  vukaFeeAmount: number;
  netAmount: number;
  currency: string;
}> {
  // Look up artist's plan slug for correct fee rate
  const artist = await prisma.artist.findUnique({
    where:  { id: params.artistId },
    select: { planSlug: true },
  });

  const plan           = getPlan(artist?.planSlug);
  const vukaFeePercent = plan.platformFeePct;
  const vukaFeeAmount  = parseFloat(((params.grossAmount * vukaFeePercent) / 100).toFixed(2));
  const netAmount      = parseFloat((params.grossAmount - vukaFeeAmount).toFixed(2));

  return {
    grossAmount: params.grossAmount,
    vukaFeePercent,
    vukaFeeAmount,
    netAmount,
    currency: params.currency || 'ZAR',
  };
}

// ── 3. EARNINGS INGESTION PIPELINE ───────────────────────────

export interface IngestionPreviewRow {
  isrc: string;
  trackId: string | null;
  trackTitle: string | null;
  artistId: string | null;
  artistName: string | null;
  streams: number;
  grossAmount: number;
  vukaFeeAmount: number;
  netAmount: number;
  currency: string;
  period: string;
  territory: string;
  matched: boolean;
  warning?: string;
}

export interface IngestionPreview {
  platformName: string;
  reportingPeriod: string;
  totalRows: number;
  matchedRows: number;
  unmatchedRows: number;
  totalGross: number;
  totalNet: number;
  currency: string;
  rows: IngestionPreviewRow[];
  warnings: string[];
}

/**
 * Parse + validate a DSP earnings CSV, return a preview for admin confirmation.
 * Does NOT write to the database — call confirmIngestion() to commit.
 */
export async function previewEarningsIngestion(params: {
  platformName: string;
  csvContent: string;
  currency?: string;
}): Promise<IngestionPreview> {
  const { platformName, csvContent } = params;
  const currency = params.currency || 'ZAR';

  const columnMap =
    DSP_COLUMN_MAPS[platformName.toLowerCase().replace(/\s+/g, '_')] ||
    DSP_COLUMN_MAPS.generic;

  const rawRows = parseCSV(csvContent);
  const warnings: string[] = [];

  const rows: IngestionPreviewRow[] = [];

  for (const raw of rawRows) {
    const isrc = (raw[columnMap.isrc] || '').trim().toUpperCase();
    if (!isrc) { warnings.push(`Row skipped — no ISRC`); continue; }

    const streamsRaw = raw[columnMap.streams] || '0';
    const grossRaw = raw[columnMap.grossAmount] || '0';
    const periodRaw = columnMap.period ? raw[columnMap.period] || '' : '';
    const territory = columnMap.territory ? raw[columnMap.territory] || '' : '';

    const streams = parseInt(streamsRaw.replace(/[^0-9]/g, ''), 10) || 0;
    const grossAmount = parseFloat(grossRaw.replace(/[^0-9.]/g, '')) || 0;
    const period = normalisePeriod(periodRaw);

    // Look up track + artist by ISRC
    // Note: we check both the DistributionRelease track model and Track model
    const distTrack = await prisma.distributionTrack.findFirst({
      where: { isrc },
      include: {
        release: {
          include: { artist: { select: { id: true, name: true } } },
        },
      },
    }).catch(() => null);

    const matched = !!distTrack;
    let trackId: string | null = null;
    let trackTitle: string | null = null;
    let artistId: string | null = null;
    let artistName: string | null = null;
    let warning: string | undefined;

    if (distTrack) {
      trackId = distTrack.id;
      trackTitle = distTrack.title;
      artistId = distTrack.release?.artist?.id || null;
      artistName = distTrack.release?.artist?.name || null;
    } else {
      warning = `ISRC ${isrc} not matched to any track — row will be logged as unmatched`;
      warnings.push(warning);
    }

    const revenueShare = artistId
      ? await calculateRevenueShare({ artistId, grossAmount, currency })
      : { grossAmount, vukaFeePercent: 15, vukaFeeAmount: +(grossAmount * 0.15).toFixed(2), netAmount: +(grossAmount * 0.85).toFixed(2), currency };

    rows.push({
      isrc,
      trackId,
      trackTitle,
      artistId,
      artistName,
      streams,
      grossAmount,
      vukaFeeAmount: revenueShare.vukaFeeAmount,
      netAmount: revenueShare.netAmount,
      currency,
      period,
      territory,
      matched,
      warning,
    });
  }

  const matchedRows = rows.filter((r) => r.matched).length;
  const totalGross = rows.reduce((s, r) => s + r.grossAmount, 0);
  const totalNet = rows.reduce((s, r) => s + r.netAmount, 0);
  const reportingPeriod = rows[0]?.period || new Date().toISOString().slice(0, 7);

  return {
    platformName,
    reportingPeriod,
    totalRows: rows.length,
    matchedRows,
    unmatchedRows: rows.length - matchedRows,
    totalGross: parseFloat(totalGross.toFixed(2)),
    totalNet: parseFloat(totalNet.toFixed(2)),
    currency,
    rows,
    warnings,
  };
}

/**
 * Commit an earnings ingestion preview to the database.
 * Creates RevenueRecord entries and credits artist balances.
 * Only matched rows are written; unmatched are logged as warnings.
 */
export async function confirmEarningsIngestion(params: {
  preview: IngestionPreview;
  adminId?: string;
}): Promise<{ created: number; skipped: number; errors: string[] }> {
  const { preview } = params;
  let created = 0;
  let skipped = 0;
  const errors: string[] = [];

  for (const row of preview.rows) {
    if (!row.matched || !row.artistId) { skipped++; continue; }

    try {
      // Deduplicate: use purchaseId as a composite idempotency key (platform:isrc:period)
      const compositeKey = `${preview.platformName}:${row.isrc}:${row.period}`;
      const existing = await prisma.revenueRecord.findFirst({
        where: { artistId: row.artistId, purchaseId: compositeKey },
      }).catch(() => null);

      if (existing) {
        skipped++;
        continue;
      }

      await prisma.revenueRecord.create({
        data: {
          artistId: row.artistId,
          type: 'distribution',
          amount: row.grossAmount,
          platformFee: row.vukaFeeAmount,
          netAmount: row.netAmount,
          currency: row.currency,
          period: row.period,
          purchaseId: compositeKey, // repurposed as idempotency key for DSP earnings rows
        },
      });

      created++;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      errors.push(`ISRC ${row.isrc}: ${msg}`);
    }
  }

  logger.info('[earnings] ingestion complete', {
    platform: preview.platformName,
    period: preview.reportingPeriod,
    created,
    skipped,
    errors: errors.length,
  });

  return { created, skipped, errors };
}

// ── 4. PAYOUT PROCESSORS ─────────────────────────────────────
// Paystack Transfers API (primary/default), PayFast Payouts API (legacy),
// Flutterwave Transfers, PayPal Payouts

// PayFast Payout (ZA artists — bank transfer via PayFast Payouts) — retained for
// accounts that still have a configured PAYFAST_API_KEY. Paystack is now the
// default/primary payout processor (see processPaystackPayout below).
export async function processPayFastPayout(params: {
  payoutRequestId: string;
  amount: number;
  currency: string;
  accountNumber: string;
  bankCode: string;
  accountHolder: string;
  reference: string;
}): Promise<{ success: boolean; referenceId?: string; error?: string }> {
  const { PAYFAST_API_KEY, PAYFAST_SANDBOX } = process.env;
  const host = PAYFAST_SANDBOX === 'true'
    ? 'https://sandbox.payfast.co.za'
    : 'https://api.payfast.co.za';

  try {
    const res = await fetch(`${host}/adhoc`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${PAYFAST_API_KEY}`,
      },
      body: JSON.stringify({
        amount:         params.amount.toFixed(2),
        currency:       params.currency || 'ZAR',
        account_number: params.accountNumber,
        bank_code:      params.bankCode,
        account_holder: params.accountHolder,
        reference:      params.reference,
        description:    'Vuka Music Royalties',
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      return { success: false, error: `PayFast API error ${res.status}: ${body}` };
    }

    const data: { token?: string; status?: string } = await res.json();
    return { success: true, referenceId: data.token };
  } catch (err: unknown) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'PayFast payout failed',
    };
  }
}

// Paystack Payout (ZA artists — Transfer Recipient + Transfer API)
// Default/primary payout processor. Creates a transfer recipient for the
// artist's bank account, then initiates a transfer for the payout amount.
export async function processPaystackPayout(params: {
  payoutRequestId: string;
  amount: number;
  currency: string;
  accountNumber: string;
  bankCode: string;
  accountHolder: string;
  reference: string;
}): Promise<{ success: boolean; referenceId?: string; error?: string }> {
  const PAYSTACK_SECRET_KEY = process.env.PAYSTACK_SECRET_KEY ?? '';
  if (!PAYSTACK_SECRET_KEY) {
    return { success: false, error: 'PAYSTACK_SECRET_KEY not configured' };
  }

  try {
    // Step 1: create (or re-use) a transfer recipient for this bank account
    const recipientRes = await fetch('https://api.paystack.co/transferrecipient', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
      },
      body: JSON.stringify({
        type:           'basa', // South African bank account
        name:           params.accountHolder,
        account_number: params.accountNumber,
        bank_code:      params.bankCode,
        currency:       params.currency || 'ZAR',
      }),
    });

    if (!recipientRes.ok) {
      const body = await recipientRes.text();
      return { success: false, error: `Paystack recipient error ${recipientRes.status}: ${body}` };
    }

    const recipientData: { data?: { recipient_code?: string } } = await recipientRes.json();
    const recipientCode = recipientData.data?.recipient_code;
    if (!recipientCode) {
      return { success: false, error: 'Paystack did not return a recipient_code' };
    }

    // Step 2: initiate the transfer
    const transferRes = await fetch('https://api.paystack.co/transfer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
      },
      body: JSON.stringify({
        source:    'balance',
        amount:    Math.round(params.amount * 100), // kobo/cents
        recipient: recipientCode,
        reason:    'Vuka Music Royalties',
        reference: params.reference,
      }),
    });

    if (!transferRes.ok) {
      const body = await transferRes.text();
      return { success: false, error: `Paystack transfer error ${transferRes.status}: ${body}` };
    }

    const transferData: { data?: { transfer_code?: string; reference?: string } } = await transferRes.json();
    return { success: true, referenceId: transferData.data?.transfer_code || transferData.data?.reference };
  } catch (err: unknown) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'Paystack payout failed',
    };
  }
}

// Flutterwave Transfer (Pan-Africa)
export async function processFlutterwavePayout(params: {
  payoutRequestId: string;
  amount: number;
  currency: string;
  accountNumber: string;
  bankCode: string;
  accountHolder: string;
  reference: string;
  country?: string;
}): Promise<{ success: boolean; referenceId?: string; error?: string }> {
  const { FLUTTERWAVE_SECRET_KEY } = process.env;
  if (!FLUTTERWAVE_SECRET_KEY) {
    return { success: false, error: 'FLUTTERWAVE_SECRET_KEY not configured' };
  }

  try {
    const res = await fetch('https://api.flutterwave.com/v3/transfers', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${FLUTTERWAVE_SECRET_KEY}`,
      },
      body: JSON.stringify({
        account_bank:   params.bankCode,
        account_number: params.accountNumber,
        amount:         params.amount,
        narration:      'Vuka Music Royalties',
        currency:       params.currency || 'ZAR',
        reference:      params.reference,
        callback_url:   `${process.env.NEXT_PUBLIC_APP_URL}/api/webhooks/flutterwave`,
        debit_currency: params.currency || 'ZAR',
        meta: {
          sender:             'Vuka Distribution',
          sender_country:     params.country || 'ZA',
          mobile_number:      '',
          recipient_address:  '',
          sender_id_number:   '',
          transfer_purpose:   'Music Royalty Payment',
        },
      }),
    });

    if (!res.ok) {
      const body = await res.text();
      return { success: false, error: `Flutterwave error ${res.status}: ${body}` };
    }

    const data: { data?: { id?: number; status?: string } } = await res.json();
    return { success: true, referenceId: String(data.data?.id || '') };
  } catch (err: unknown) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'Flutterwave transfer failed',
    };
  }
}

// PayPal Payouts (International)
export async function processPayPalPayout(params: {
  payoutRequestId: string;
  amount: number;
  currency: string;
  paypalEmail: string;
  reference: string;
}): Promise<{ success: boolean; referenceId?: string; error?: string }> {
  const { PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, PAYPAL_SANDBOX } = process.env;
  if (!PAYPAL_CLIENT_ID || !PAYPAL_CLIENT_SECRET) {
    return { success: false, error: 'PayPal credentials not configured' };
  }

  const host = PAYPAL_SANDBOX === 'true'
    ? 'https://api-m.sandbox.paypal.com'
    : 'https://api-m.paypal.com';

  try {
    // Get access token
    const tokenRes = await fetch(`${host}/v1/oauth2/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${Buffer.from(`${PAYPAL_CLIENT_ID}:${PAYPAL_CLIENT_SECRET}`).toString('base64')}`,
      },
      body: 'grant_type=client_credentials',
    });

    if (!tokenRes.ok) {
      return { success: false, error: 'PayPal auth failed' };
    }

    const { access_token }: { access_token: string } = await tokenRes.json();

    // Send payout
    const payoutRes = await fetch(`${host}/v1/payments/payouts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${access_token}`,
      },
      body: JSON.stringify({
        sender_batch_header: {
          sender_batch_id: params.reference,
          email_subject:   'Vuka Royalty Payment',
          email_message:   'You have received a royalty payment from Vuka Music.',
        },
        items: [{
          recipient_type: 'EMAIL',
          amount: {
            value:    params.amount.toFixed(2),
            currency: params.currency || 'USD',
          },
          receiver:  params.paypalEmail,
          note:      'Vuka Music Royalties',
          sender_item_id: params.payoutRequestId,
        }],
      }),
    });

    if (!payoutRes.ok) {
      const body = await payoutRes.text();
      return { success: false, error: `PayPal payout error ${payoutRes.status}: ${body}` };
    }

    const data: { batch_header?: { payout_batch_id?: string } } = await payoutRes.json();
    return { success: true, referenceId: data.batch_header?.payout_batch_id };
  } catch (err: unknown) {
    return {
      success: false,
      error: err instanceof Error ? err.message : 'PayPal payout failed',
    };
  }
}

// ── 5. PAYOUT DISPATCHER ─────────────────────────────────────
// Routes an approved PayoutRequest to the correct processor.

export type PayoutMethod = 'paystack' | 'payfast' | 'flutterwave' | 'paypal' | 'bank_transfer';

export async function dispatchPayout(payoutRequestId: string): Promise<{
  success: boolean;
  referenceId?: string;
  error?: string;
}> {
  const request = await prisma.payoutRequest.findUnique({
    where: { id: payoutRequestId },
    include: {
      bankAccount: true,
      artist: { include: { user: { select: { email: true } } } },
    },
  });

  if (!request) return { success: false, error: 'Payout request not found' };
  if (request.status !== 'approved') {
    return { success: false, error: `Cannot dispatch: status is ${request.status}` };
  }

  // Mark as processing
  await prisma.payoutRequest.update({
    where: { id: payoutRequestId },
    data: { status: 'processing' },
  });

  const reference = `VUKA-${payoutRequestId.slice(-8).toUpperCase()}`;
  let result: { success: boolean; referenceId?: string; error?: string };

  const method = (request.bankAccount?.accountType || 'bank_transfer') as PayoutMethod;

  if (method === 'paypal') {
    const paypalEmail = request.artist.user?.email || '';
    result = await processPayPalPayout({
      payoutRequestId,
      amount: request.amount,
      currency: request.currency,
      paypalEmail,
      reference,
    });
  } else if (method === 'flutterwave') {
    const ba = request.bankAccount;
    result = await processFlutterwavePayout({
      payoutRequestId,
      amount: request.amount,
      currency: request.currency,
      accountNumber: ba?.accountNumber || '',
      bankCode: ba?.branchCode || '',
      accountHolder: ba?.accountHolder || '',
      reference,
      country: 'ZA',
    });
  } else if (method === 'payfast') {
    // Explicit PayFast — only used for accounts still configured with PAYFAST_API_KEY
    const ba = request.bankAccount;
    result = await processPayFastPayout({
      payoutRequestId,
      amount: request.amount,
      currency: request.currency,
      accountNumber: ba?.accountNumber || '',
      bankCode: ba?.branchCode || '',
      accountHolder: ba?.accountHolder || '',
      reference,
    });
  } else {
    // paystack or bank_transfer — default to Paystack Transfers API
    const ba = request.bankAccount;
    result = await processPaystackPayout({
      payoutRequestId,
      amount: request.amount,
      currency: request.currency,
      accountNumber: ba?.accountNumber || '',
      bankCode: ba?.branchCode || '',
      accountHolder: ba?.accountHolder || '',
      reference,
    });
  }

  if (result.success) {
    await prisma.payoutRequest.update({
      where: { id: payoutRequestId },
      data: {
        status: 'processing',
        adminNotes: `Dispatched via ${method} — ref: ${result.referenceId || reference}`,
        processedAt: new Date(),
      },
    });
  } else {
    await prisma.payoutRequest.update({
      where: { id: payoutRequestId },
      data: {
        status: 'rejected',
        adminNotes: `Dispatch failed: ${result.error}`,
      },
    });
  }

  return result;
}

// ── 6. WEBHOOK STATUS HANDLERS ────────────────────────────────

export async function handleFlutterwaveWebhook(payload: {
  event: string;
  data: {
    id: string;
    reference: string;
    amount: number;
    currency: string;
    status: string;
    complete_message?: string;
  };
}): Promise<void> {
  const { event, data } = payload;
  if (event !== 'transfer.completed' && event !== 'transfer.failed') return;

  // Extract payoutRequestId from reference: VUKA-XXXXXXXX
  const refSuffix = data.reference?.replace('VUKA-', '');
  if (!refSuffix) return;

  const request = await prisma.payoutRequest.findFirst({
    where: { id: { endsWith: refSuffix.toLowerCase() } },
  });
  if (!request) return;

  const newStatus = data.status === 'SUCCESSFUL' ? 'paid' : 'rejected';

  await prisma.payoutRequest.update({
    where: { id: request.id },
    data: {
      status: newStatus,
      processedAt: new Date(),
      adminNotes: `Flutterwave webhook: ${data.status} — ${data.complete_message || ''}`,
    },
  });

  if (newStatus === 'paid') {
    await prisma.artistPayout.create({
      data: {
        artistId:    request.artistId,
        amount:      request.amount,
        currency:    request.currency,
        status:      'paid',
        method:      'flutterwave',
        reference:   data.reference,
        notes:       `PayoutRequest ${request.id}`,
        processedAt: new Date(),
      },
    });
  }
}

export async function handlePayPalWebhook(payload: {
  event_type: string;
  resource: {
    batch_header?: {
      payout_batch_id: string;
      batch_status: string;
      sender_batch_header: { sender_batch_id: string };
    };
    payout_item_id?: string;
    transaction_status?: string;
  };
}): Promise<void> {
  const { event_type, resource } = payload;
  if (!event_type.startsWith('PAYMENT.PAYOUTS-ITEM')) return;

  const senderBatchId = resource.batch_header?.sender_batch_header?.sender_batch_id || '';
  const refSuffix = senderBatchId.replace('VUKA-', '');
  if (!refSuffix) return;

  const request = await prisma.payoutRequest.findFirst({
    where: { id: { endsWith: refSuffix.toLowerCase() } },
  });
  if (!request) return;

  const success = event_type === 'PAYMENT.PAYOUTS-ITEM.SUCCEEDED';
  const failed = event_type === 'PAYMENT.PAYOUTS-ITEM.FAILED' || event_type === 'PAYMENT.PAYOUTS-ITEM.RETURNED';

  if (success) {
    await prisma.payoutRequest.update({
      where: { id: request.id },
      data: { status: 'paid', processedAt: new Date(), adminNotes: `PayPal: ${event_type}` },
    });
    await prisma.artistPayout.create({
      data: {
        artistId:    request.artistId,
        amount:      request.amount,
        currency:    request.currency,
        status:      'paid',
        method:      'paypal',
        reference:   senderBatchId,
        notes:       `PayoutRequest ${request.id}`,
        processedAt: new Date(),
      },
    });
  } else if (failed) {
    await prisma.payoutRequest.update({
      where: { id: request.id },
      data: { status: 'rejected', adminNotes: `PayPal: ${event_type}` },
    });
  }
}
