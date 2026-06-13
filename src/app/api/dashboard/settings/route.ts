export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireArtist } from '@/lib/auth';
import prisma from '@/lib/prisma';

export async function GET() {
  try {
    const user = await requireArtist();
    if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    const artist = await prisma.artist.findUnique({
      where: { id: user.artist.id },
      include: { user: { select: { email: true, name: true } } },
    });
    return NextResponse.json({ artist });
  } catch (err) {
    console.error('[settings] GET error:', err);
    return NextResponse.json({ error: 'Database error', artist: null }, { status: 503 });
  }
}

export async function PATCH(req: NextRequest) {
  try {
    const user = await requireArtist();
    if (!user?.artist) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    const body = await req.json();
    const { name, bio, city, country, genreTags, photoUrl, coverUrl, socialLinks, currency, paystackRecipient } = body;
    const artist = await prisma.artist.update({
      where: { id: user.artist.id },
      data: {
        name: name || undefined,
        bio: bio !== undefined ? bio : undefined,
        city: city !== undefined ? city : undefined,
        country: country || undefined,
        genreTags: genreTags || undefined,
        photoUrl: photoUrl !== undefined ? photoUrl : undefined,
        coverUrl: coverUrl !== undefined ? coverUrl : undefined,
        socialLinks: socialLinks ? JSON.parse(JSON.stringify(socialLinks)) : undefined,
        currency: currency || undefined,
        // Allow saving empty string to clear, or a real value
        ...(paystackRecipient !== undefined && { paystackRecipient: paystackRecipient.trim() || null }),
      },
    });
    return NextResponse.json({ artist });
  } catch (err) {
    console.error('[settings] PATCH error:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 503 });
  }
}
