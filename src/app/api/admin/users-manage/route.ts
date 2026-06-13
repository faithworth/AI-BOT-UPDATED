/**
 * GET  /api/admin/users-manage?q=&role=all&page=1
 * POST /api/admin/users-manage { userId, action, value?, reason?, months? }
 * Actions: suspend | unsuspend | set_role | verify | unverify | delete | set_plan
 */
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import prisma, { queryRaw, executeRaw } from '@/lib/prisma';
import { auditLog } from '@/lib/audit';
import { sendAccountSuspended } from '@/lib/emails';

const APP_URL = () => process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';

export async function GET(req: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const q     = searchParams.get('q')    || '';
  const role  = searchParams.get('role') || 'all';
  const page  = Math.max(1, parseInt(searchParams.get('page') || '1'));
  const limit = 50;
  const offset = (page - 1) * limit;

  try {
    const conditions: string[] = [];
    const params: any[]        = [];
    let   pi                   = 1;

    if (q) {
      conditions.push(`(u.name ILIKE $${pi} OR u.email ILIKE $${pi})`);
      params.push(`%${q}%`);
      pi++;
    }
    if (role !== 'all') {
      conditions.push(`u.role = $${pi}`);
      params.push(role.toLowerCase());
      pi++;
    }

    const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';

    const [rows, countRows] = await Promise.all([
      queryRaw<any>(`
        SELECT
          u.id, u.name, u.email, u.role, u."createdAt", u."isSuspended",
          a.id             AS "artistId",
          a.slug           AS "artistSlug",
          a."isVerified",
          a."paystackRecipient",
          a."totalPlays",
          a."planSlug",
          a."planExpiresAt",
          (SELECT COUNT(*) FROM "Purchase" p WHERE p."userId" = u.id)::int AS purchases
        FROM "User" u
        LEFT JOIN "Artist" a ON a."userId" = u.id
        ${where}
        ORDER BY u."createdAt" DESC
        LIMIT ${limit} OFFSET ${offset}
      `, ...params),
      queryRaw<any>(`
        SELECT COUNT(*)::int AS total FROM "User" u ${where}
      `, ...params),
    ]);

    const total = countRows[0]?.total ?? 0;

    const users = rows.map(r => ({
      id: r.id, name: r.name, email: r.email, role: r.role,
      createdAt: r.createdAt, isSuspended: r.isSuspended,
      artist: r.artistId ? {
        id:              r.artistId,
        slug:            r.artistSlug,
        isVerified:      r.isVerified,
        paystackRecipient: r.paystackRecipient,
        totalPlays:      r.totalPlays ?? 0,
        planSlug:        r.planSlug ?? 'free',
        planExpiresAt:   r.planExpiresAt ?? null,
      } : null,
      _count: { purchases: r.purchases ?? 0 },
    }));

    return NextResponse.json({ users, total, page, pages: Math.ceil(total / limit) });
  } catch (err) {
    console.error('[admin/users-manage] GET error:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 503 });
  }
}

export async function POST(req: NextRequest) {
  const admin = await requireAdmin();
  if (!admin) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const { userId, action, value, reason, months } = await req.json();
    if (!userId || !action)
      return NextResponse.json({ error: 'userId and action required' }, { status: 400 });

    const target = await prisma.user.findUnique({
      where: { id: userId },
      include: { artist: { select: { id: true, slug: true, isVerified: true } } },
    });
    if (!target) return NextResponse.json({ error: 'User not found' }, { status: 404 });

    const elevated = ['admin', 'owner', 'super_admin'];
    if (elevated.includes(target.role) && !['owner', 'super_admin'].includes(admin.role))
      return NextResponse.json({ error: 'Cannot modify another admin' }, { status: 403 });

    switch (action) {
      case 'suspend': {
        await prisma.user.update({
          where: { id: userId },
          data: { isSuspended: true, suspendedAt: new Date(), suspendedReason: reason || 'Suspended by admin' },
        });
        await auditLog.adminAction('auth.ban', 'User', userId, admin.id, reason || '');
        try {
          await sendAccountSuspended({
            to: target.email,
            displayName: target.name || target.email,
            reason: reason || "Your account has been suspended for violating Vuka's Terms of Service.",
            appealUrl: `${APP_URL()}/appeal?userId=${userId}`,
          });
        } catch (e) { console.error('[admin/users] suspend email failed:', e); }
        return NextResponse.json({ ok: true });
      }

      case 'unsuspend': {
        await prisma.user.update({ where: { id: userId }, data: { isSuspended: false, suspendedReason: '' } });
        await auditLog.adminAction('auth.unban', 'User', userId, admin.id, '');
        return NextResponse.json({ ok: true });
      }

      case 'set_role': {
        const allowed = ['fan', 'artist', 'producer', 'industry', 'moderator', 'admin'];
        if (!allowed.includes(value))
          return NextResponse.json({ error: 'Invalid role' }, { status: 400 });
        const prev = target.role;
        await prisma.user.update({ where: { id: userId }, data: { role: value } });
        await auditLog.adminAction('auth.role_change', 'User', userId, admin.id, `${prev}→${value}`);
        return NextResponse.json({ ok: true });
      }

      case 'verify': {
        const a = target.artist ?? await prisma.artist.findUnique({ where: { userId } });
        if (!a) return NextResponse.json({ error: 'User has no artist profile' }, { status: 400 });
        await prisma.artist.update({ where: { id: a.id }, data: { isVerified: true } });
        await auditLog.adminAction('moderation.artist_verified', 'Artist', a.id, admin.id, 'verified');
        return NextResponse.json({ ok: true });
      }

      case 'unverify': {
        const a = target.artist ?? await prisma.artist.findUnique({ where: { userId } });
        if (!a) return NextResponse.json({ error: 'User has no artist profile' }, { status: 400 });
        await prisma.artist.update({ where: { id: a.id }, data: { isVerified: false } });
        await auditLog.adminAction('moderation.artist_verified', 'Artist', a.id, admin.id, 'unverified');
        return NextResponse.json({ ok: true });
      }

      case 'delete': {
        await prisma.user.delete({ where: { id: userId } });
        await auditLog.adminAction('admin.user_deleted', 'User', userId, admin.id, reason || '');
        return NextResponse.json({ ok: true });
      }

      case 'set_plan': {
        const allowedPlans = ['free', 'pro', 'label'];
        if (!allowedPlans.includes(value))
          return NextResponse.json({ error: 'Invalid plan' }, { status: 400 });

        const artistForPlan = target.artist ?? await prisma.artist.findUnique({ where: { userId } });
        if (!artistForPlan)
          return NextResponse.json({ error: 'User has no artist profile' }, { status: 400 });

        const { PLANS } = await import('@/lib/plans');
        const plan = PLANS.find(p => p.slug === value);
        if (!plan) return NextResponse.json({ error: 'Plan not found' }, { status: 400 });

        const now = new Date();
        let expiresAt: Date | null = null;
        if (plan.priceZAR > 0 && months && months > 0) {
          expiresAt = new Date(now);
          expiresAt.setMonth(expiresAt.getMonth() + months);
        }

        await prisma.$executeRaw`
          UPDATE "Artist" SET "planSlug" = ${value}, "planExpiresAt" = ${expiresAt}
          WHERE id = ${artistForPlan.id}
        `;
        await auditLog.adminAction('plan.admin_override', 'Artist', artistForPlan.id, admin.id,
          `Plan set to ${value} (expires ${expiresAt?.toISOString() ?? 'never'})`);
        return NextResponse.json({ ok: true });
      }

      default:
        return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (err: any) {
    console.error('[admin/users-manage] POST error:', err?.message);
    return NextResponse.json({ error: 'Action failed' }, { status: 503 });
  }
}
