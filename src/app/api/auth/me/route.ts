// src/app/api/auth/me/route.ts
// Returns the authenticated user's profile, role, and platform flags.
// DB is the SINGLE SOURCE OF TRUTH for role — never URL params, never session claims.
// ADMIN_EMAIL env var automatically elevates that user to OWNER on first hit.
//
// Fixes applied:
//   - IndustryUser.profession → .role (actual field name in schema)
//   - IndustryUser.isVerified → .verified (actual field name in schema)
//   - User.isSuspended now in schema (added via migration)
//   - requireAdmin() in auth.ts now includes owner/super_admin — this route
//     already handled elevation correctly

export const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server';
import { getServerUser } from '@/lib/auth';
import prisma from '@/lib/prisma';

const ADMIN_EMAIL = (process.env.ADMIN_EMAIL ?? '').toLowerCase().trim();

export async function GET() {
  try {
    const authUser = await getServerUser();
    if (!authUser) {
      // getServerUser returns null either when no session or when user not in DB.
      // Check if there's a Supabase session even though no DB user — can happen for admin.
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Fetch from DB — always fresh, never cached
    let dbUser = await prisma.user.findUnique({
      where: { id: authUser.id },
      include: {
        artist: {
          select: {
            id: true,
            name: true,
            slug: true,
            photoUrl: true,
            paystackRecipient: true,
            isVerified: true,
          },
        },
        industryUser: {
          // IndustryUser schema fields: id, companyName, role, website, verified
          // 'profession' does not exist — the field is 'role'
          // 'isVerified' does not exist — the field is 'verified'
          select: { id: true, role: true, verified: true },
        },
      },
    });

    if (!dbUser) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    // ADMIN_EMAIL elevation — if this user's email matches the env var,
    // and they're not already owner/super_admin, elevate them now.
    if (
      ADMIN_EMAIL &&
      dbUser.email?.toLowerCase() === ADMIN_EMAIL &&
      !['owner', 'super_admin'].includes(dbUser.role ?? '')
    ) {
      dbUser = await prisma.user.update({
        where: { id: dbUser.id },
        data: { role: 'owner' },
        include: {
          artist: {
            select: {
              id: true,
              name: true,
              slug: true,
              photoUrl: true,
              paystackRecipient: true,
              isVerified: true,
            },
          },
          industryUser: {
            select: { id: true, role: true, verified: true },
          },
        },
      });
      console.log(`[auth/me] ADMIN_EMAIL elevation: ${dbUser.email} → owner`);
    }

    const role = dbUser.role ?? 'fan';
    const isAdmin = ['owner', 'super_admin', 'admin', 'moderator'].includes(role);

    return NextResponse.json({
      id: dbUser.id,
      email: dbUser.email,
      name: dbUser.name,
      role,
      isAdmin,
      isArtist: !!dbUser.artist,
      isIndustry: !!dbUser.industryUser,
      isFan: !dbUser.artist && !dbUser.industryUser && !isAdmin,
      artist: dbUser.artist ?? null,
      industryUser: dbUser.industryUser ?? null,
      isSuspended: dbUser.isSuspended ?? false,
    });
  } catch (err) {
    console.error('[auth/me]', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
