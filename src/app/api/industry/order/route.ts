// src/app/api/industry/order/route.ts
// Artist pays for an industry service through Vuka.
// POST  — create order + Paystack payment URL
// GET   — artist views their own industry orders

export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { getServerUser } from '@/lib/auth';
import prisma, { queryRaw, executeRaw } from '@/lib/prisma';
import { initializeTransaction, generateReference } from '@/lib/paystack';

// 10% platform fee charged to industry (deducted from what industry receives)
const INDUSTRY_PLATFORM_FEE_PCT = 0.10;

function calcFees(amount: number) {
  const platformFee = Math.round(amount * INDUSTRY_PLATFORM_FEE_PCT * 100) / 100;
  const netAmount   = Math.round((amount - platformFee) * 100) / 100;
  return { platformFee, netAmount };
}

// POST — initiate payment for an industry service
export async function POST(req: NextRequest) {
  try {
    const user = await getServerUser();
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const artist = await prisma.artist.findUnique({ where: { userId: user.id } });
    if (!artist) return NextResponse.json({ error: 'Artist profile required to order services' }, { status: 403 });

    const { serviceId, requirements } = await req.json();
    if (!serviceId) return NextResponse.json({ error: 'serviceId required' }, { status: 400 });

    const service = await prisma.industryService.findUnique({
      where: { id: serviceId },
      include: { industryUser: { include: { user: { select: { id: true, name: true, email: true } } } } },
    });
    if (!service || !service.isActive) {
      return NextResponse.json({ error: 'Service not found or inactive' }, { status: 404 });
    }

    // Can't order your own service (industry user ordering their own listing)
    if (service.industryUser.userId === user.id) {
      return NextResponse.json({ error: 'Cannot order your own service' }, { status: 400 });
    }

    const amount = service.priceZAR;
    if (!amount || amount <= 0) {
      return NextResponse.json({ error: 'Invalid service price' }, { status: 400 });
    }

    const { platformFee, netAmount } = calcFees(amount);

    // Create pending order
    const order = await queryRaw(
      `INSERT INTO "IndustryServiceOrder"
         (id, "serviceId", "artistId", "industryUserId", amount, "platformFee", "netAmount", currency, status, requirements, "createdAt", "updatedAt")
       VALUES (gen_random_uuid()::text, $1, $2, $3, $4, $5, $6, 'ZAR', 'pending', $7, now(), now())
       RETURNING *`,
      serviceId,
      artist.id,
      service.industryUserId,
      amount,
      platformFee,
      netAmount,
      requirements?.trim() || '',
    );
    const o = order[0];

    // Initialize Paystack transaction (artist pays the full amount; platform takes 10% from industry's net)
    const appUrl    = process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';
    const reference = generateReference('ISO');

    const result = await initializeTransaction({
      email:       user.email,
      amountZAR:   amount,
      reference,
      callbackUrl: `${appUrl}/dashboard/purchases?iso=${o.id}&success=1`,
      metadata: {
        orderId:   o.id,
        type:      'industry_order',
        serviceTitle: `${service.title} — ${service.industryUser.user.name || 'Professional'}`.slice(0, 100),
      },
    });

    // Store reference so the order can be reconciled later
    await executeRaw(
      `UPDATE "IndustryServiceOrder" SET "payfastPaymentId" = $1, "updatedAt" = now() WHERE id = $2`,
      reference,
      o.id,
    );

    return NextResponse.json({ ok: true, orderId: o.id, payUrl: result.authorizationUrl, reference, amount, platformFee, netAmount });
  } catch (err) {
    console.error('[industry/order POST]', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}

// GET — list orders for the logged-in artist or industry user
export async function GET() {
  try {
    const user = await getServerUser();
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    if (user.role === 'industry') {
      const iu = await prisma.industryUser.findUnique({ where: { userId: user.id } });
      if (!iu) return NextResponse.json({ orders: [] });

      const orders = await queryRaw(
        `SELECT iso.*, 
                s.title AS "serviceTitle", s.category, s."priceZAR",
                a.name AS "artistName", a.slug AS "artistSlug", a."photoUrl" AS "artistPhoto"
           FROM "IndustryServiceOrder" iso
           JOIN "IndustryService" s ON s.id = iso."serviceId"
           JOIN "Artist" a ON a.id = iso."artistId"
          WHERE iso."industryUserId" = $1
          ORDER BY iso."createdAt" DESC`,
        iu.id,
      );
      return NextResponse.json({ orders });
    }

    // Artist view
    const artist = await prisma.artist.findUnique({ where: { userId: user.id } });
    if (!artist) return NextResponse.json({ orders: [] });

    const orders = await queryRaw(
      `SELECT iso.*,
              s.title AS "serviceTitle", s.category, s."priceZAR",
              iu."companyName", iu.role AS "industryRole",
              u.name AS "providerName"
         FROM "IndustryServiceOrder" iso
         JOIN "IndustryService" s ON s.id = iso."serviceId"
         JOIN "IndustryUser" iu ON iu.id = iso."industryUserId"
         JOIN "User" u ON u.id = iu."userId"
        WHERE iso."artistId" = $1
        ORDER BY iso."createdAt" DESC`,
      artist.id,
    );
    return NextResponse.json({ orders });
  } catch (err) {
    console.error('[industry/order GET]', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
