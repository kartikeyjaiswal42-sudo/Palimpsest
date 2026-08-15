/**
 * A Durable Object holding the two counters an open endpoint needs.
 *
 * The account's Workers AI allowance is 10,000 neurons a day and one analysis costs about
 * 2.5, so roughly 4,000 analyses. That is plenty for a demonstration and trivially drainable
 * by anyone with a loop, which is exactly why the original `observer-worker/` sat behind a
 * shared secret. This deployment is meant to be opened without one, so the protection has to
 * live here instead:
 *
 *   1. a per-IP sliding window, so one client cannot monopolise the service;
 *   2. a global daily neuron budget, so the day's allowance cannot be spent in an hour.
 *
 * A Durable Object rather than KV because both counters are read-modify-write on a hot key.
 * KV is eventually consistent, so concurrent requests would each read a stale count and the
 * limit would be advisory at best; KV's free write allowance (1,000/day) is also *below* the
 * neuron budget being protected, which would make storage the binding constraint rather than
 * the thing it is guarding.
 *
 * Failure is deliberately closed for the budget and open for the rate limit: if this object
 * is unreachable, a burst of requests is a nuisance while an unmetered spend is not.
 */

const DAY_MS = 86_400_000;

export class Budget {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.perMinute = Number(env.RATE_PER_MINUTE ?? 6);
    this.perHour = Number(env.RATE_PER_HOUR ?? 40);
    this.dailyNeurons = Number(env.DAILY_NEURON_BUDGET ?? 7000);
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === '/reserve') return this.reserve(await request.json());
    if (url.pathname === '/settle') return this.settle(await request.json());
    if (url.pathname === '/state') return Response.json(await this.snapshot());
    return new Response('not found', { status: 404 });
  }

  async snapshot() {
    const day = new Date().toISOString().slice(0, 10);
    const spent = (await this.state.storage.get(`neurons:${day}`)) ?? 0;
    const calls = (await this.state.storage.get(`calls:${day}`)) ?? 0;
    return {
      day,
      neuronsSpent: Math.round(spent * 100) / 100,
      dailyNeuronBudget: this.dailyNeurons,
      analysesToday: calls,
      perMinute: this.perMinute,
      perHour: this.perHour,
    };
  }

  /** Check both limits and, if they pass, record the attempt. */
  async reserve({ ip, estimate = 3 }) {
    const now = Date.now();
    const day = new Date(now).toISOString().slice(0, 10);

    const spent = (await this.state.storage.get(`neurons:${day}`)) ?? 0;
    if (spent + estimate > this.dailyNeurons) {
      return Response.json({
        ok: false,
        reason: 'budget',
        detail:
          "This demonstration's daily budget on Cloudflare Workers AI is spent. It resets at " +
          '00:00 UTC. The detector is not down and nothing is wrong with your essay — the ' +
          'observer is a 30B model on a metered free allowance, and refusing is more honest ' +
          'than degrading to a smaller one whose numbers would mean something different.',
        retryAfter: Math.ceil((Date.parse(`${day}T23:59:59Z`) + 1000 - now) / 1000),
      });
    }

    const key = `ip:${ip}`;
    const hits = ((await this.state.storage.get(key)) ?? []).filter((t) => now - t < 3_600_000);
    const minuteHits = hits.filter((t) => now - t < 60_000);
    const overMinute = minuteHits.length >= this.perMinute;
    if (overMinute || hits.length >= this.perHour) {
      const oldest = overMinute ? minuteHits[0] : hits[0];
      const window = overMinute ? 60_000 : 3_600_000;
      const retryAfter = Math.max(1, Math.ceil((oldest + window - now) / 1000));
      const wait = retryAfter < 90
        ? `${retryAfter} seconds`
        : `${Math.ceil(retryAfter / 60)} minutes`;
      return Response.json({
        ok: false,
        reason: 'rate',
        detail:
          `Too many analyses from this address (limit ${this.perMinute} a minute, ` +
          `${this.perHour} an hour). Each one runs a 30B model over your essay on a shared ` +
          `free allowance. Try again in about ${wait}.`,
        retryAfter,
      });
    }

    hits.push(now);
    await this.state.storage.put(key, hits);
    // Charge the estimate up front so concurrent requests cannot all pass the budget check;
    // `settle` corrects it to the real cost once the observer reports back.
    await this.state.storage.put(`neurons:${day}`, spent + estimate);
    await this.state.storage.put(`calls:${day}`, ((await this.state.storage.get(`calls:${day}`)) ?? 0) + 1);
    // Only arm the cleanup alarm if one is not already pending. `setAlarm` REPLACES the
    // scheduled time rather than adding to it, so calling it on every reservation pushed the
    // alarm 24 hours into the future on each request -- meaning that under any continuous
    // traffic at all the cleanup this object schedules could never fire, and the per-IP
    // history and old day counters it exists to drop would accumulate indefinitely. The
    // alarm is a floor on cleanup, not a per-request timer.
    if ((await this.state.storage.getAlarm()) === null) {
      await this.state.storage.setAlarm(now + DAY_MS);
    }
    return Response.json({ ok: true, estimate });
  }

  /** Replace the reserved estimate with what the call actually cost. */
  async settle({ estimate = 3, neurons = null }) {
    if (typeof neurons !== 'number') return Response.json({ ok: true, adjusted: 0 });
    const day = new Date().toISOString().slice(0, 10);
    const spent = (await this.state.storage.get(`neurons:${day}`)) ?? 0;
    const adjusted = Math.max(0, spent - estimate + neurons);
    await this.state.storage.put(`neurons:${day}`, adjusted);
    return Response.json({ ok: true, adjusted });
  }

  /** Drop per-IP history and counters older than a day so storage does not grow forever. */
  async alarm() {
    const now = Date.now();
    const keep = new Date(now).toISOString().slice(0, 10);
    const yesterday = new Date(now - DAY_MS).toISOString().slice(0, 10);
    const all = await this.state.storage.list();
    for (const [key, value] of all) {
      if (key.startsWith('ip:')) {
        const live = (value ?? []).filter((t) => now - t < 3_600_000);
        if (live.length) await this.state.storage.put(key, live);
        else await this.state.storage.delete(key);
      } else if (!key.endsWith(keep) && !key.endsWith(yesterday)) {
        await this.state.storage.delete(key);
      }
    }
  }
}
