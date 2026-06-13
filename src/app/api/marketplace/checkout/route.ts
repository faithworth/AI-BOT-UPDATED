// src/app/api/marketplace/checkout/route.ts
// Initiate Paystack payment for a marketplace service order.
// On charge.success → /api/marketplace/checkout/notify activates the order.

export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';
import { initializeTransaction, generateReference } from '@/lib/paystack';
import { requireAuth } from '@/lib/auth';

export async function POST(req: NextRequest) {
  try {
    const { serviceId, packageName, amount, requirements, buyerName, buyerEmail } = await req.json();

    if (!serviceId || !amount || !buyerEmail || !buyerName) {
      return NextResponse.json({ error: 'serviceId, amount, buyerName and buyerEmail are required' }, { status: 400 });
    }

    const service = await prisma.marketplaceService.findUnique({
      where: { id: serviceId },
      include: { artist: { include: { user: true } } },
    });
    if (!service?.isActive) return NextResponse.json({ error: 'Service not available' }, { status: 404 });

    const user = await requireAuth();
    if (!user) return NextResponse.json({ error: 'You must be logged in to place an order' }, { status: 401 });

    const buyerArtist = await prisma.artist.findUnique({ where: { userId: user.id }, select: { id: true } });
    if (buyerArtist?.id === service.artistId) return NextResponse.json({ error: 'Cannot order your own service' }, { status: 400 });

    const deadline = new Date();
    deadline.setDate(deadline.getDate() + (service.deliveryDays || 7));

    const order = await prisma.marketplaceOrder.create({
      data: {
        serviceId,
        buyerId:      user.id,
        sellerId:     service.artistId,
        amount:       Number(amount),
        currency:     'ZAR',
        requirements: requirements || '',
        status:       'pending',
        deadline,
      },
    });

    const appUrl    = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const reference = generateReference('MKT');

    const result = await initializeTransaction({
      email:       buyerEmail || user.email,
      amountZAR:   Number(amount),
      reference,
      callbackUrl: `${appUrl}/marketplace?order=success&id=${order.id}`,
      metadata: {
        orderId:   order.id,
        serviceId,
        artistId:  service.artistId,
        buyerEmail: buyerEmail || user.email,
        type:      'marketplace',
      },
    });

    // Store reference on order for webhook lookup
    await prisma.marketplaceOrder.update({
      where: { id: order.id },
      data:  { paystackReference: reference },
    });

    return NextResponse.json({
      authorizationUrl: result.authorizationUrl,
      method:           'paystack',
      orderId:          order.id,
    });
  } catch (err: any) {
    console.error('[marketplace/checkout] error:', err?.message);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
