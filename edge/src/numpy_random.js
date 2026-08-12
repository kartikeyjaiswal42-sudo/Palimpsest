/**
 * `np.random.default_rng(seed).integers(0, n, size=n)`, reproduced exactly.
 *
 * Why bother. `detect/document.py` reports the machine share with a bootstrap interval, and
 * that interval is seeded (`rng_seed=0`) so the same essay always yields the same answer. A
 * different generator here would give a *statistically* equivalent interval that prints
 * different digits, which would mean the deployed application quietly disagrees with the
 * numbers the Python build produces and with every figure in the evaluation docs. The
 * interval is small and cheap to reproduce properly, so it is reproduced properly.
 *
 * Three pieces, all from NumPy's own source:
 *   1. SeedSequence's entropy mixer, which turns the integer seed into 128 bits of state.
 *   2. PCG64 (XSL RR 128/64), whose 128-bit LCG needs BigInt.
 *   3. `Generator.integers`, which is Lemire's bounded method with rejection — *not* the
 *      masked rejection RandomState uses, and not modulo.
 *
 * test/parity.test.mjs checks the emitted sequence against NumPy rather than trusting this.
 */

const U32 = 0xffffffff;
const MASK64 = (1n << 64n) - 1n;
const MASK128 = (1n << 128n) - 1n;

const INIT_A = 0x43b0d7e5;
const MULT_A = 0x931e8875;
const INIT_B = 0x8b51f9dd;
const MULT_B = 0x58f38ded;
const MIX_MULT_L = 0xca01f9dd;
const MIX_MULT_R = 0x4973f715;
const XSHIFT = 16;
const POOL_SIZE = 4;

const PCG_MULT = 0x2360ed051fc65da44385df649fccf645n;

/** uint32 multiply without losing the low bits to float64 rounding. */
const mul32 = (a, b) => Math.imul(a, b) >>> 0;

class Hasher {
  constructor(init) {
    this.hashConst = init >>> 0;
  }

  mix(value) {
    let v = (value ^ this.hashConst) >>> 0;
    this.hashConst = mul32(this.hashConst, MULT_A);
    v = mul32(v, this.hashConst);
    v = (v ^ (v >>> XSHIFT)) >>> 0;
    return v;
  }
}

const mix2 = (x, y) => {
  let result = (mul32(MIX_MULT_L, x) - mul32(MIX_MULT_R, y)) >>> 0;
  result = (result ^ (result >>> XSHIFT)) >>> 0;
  return result;
};

/** `SeedSequence(entropy).pool` for a small non-negative integer seed. */
function seedPool(entropy) {
  if (!Number.isInteger(entropy) || entropy < 0 || entropy > U32) {
    throw new Error('only small non-negative integer seeds are supported');
  }
  // `_int_to_uint32_array` emits one word per 32 bits and, as a special case, a single
  // zero word for 0 -- not an empty array. The spawn key is empty for a bare seed.
  const entropyArray = [entropy >>> 0];
  const hasher = new Hasher(INIT_A);
  const pool = new Uint32Array(POOL_SIZE);
  for (let i = 0; i < POOL_SIZE; i += 1) {
    pool[i] = hasher.mix(i < entropyArray.length ? entropyArray[i] : 0);
  }
  for (let src = 0; src < POOL_SIZE; src += 1) {
    for (let dst = 0; dst < POOL_SIZE; dst += 1) {
      if (src !== dst) pool[dst] = mix2(pool[dst], hasher.mix(pool[src]));
    }
  }
  for (let src = POOL_SIZE; src < entropyArray.length; src += 1) {
    for (let dst = 0; dst < POOL_SIZE; dst += 1) {
      pool[dst] = mix2(pool[dst], hasher.mix(entropyArray[src]));
    }
  }
  return pool;
}

/** `SeedSequence.generate_state(nWords, uint32)`. */
function generateState(pool, nWords) {
  const hasher = { c: INIT_B >>> 0 };
  const out = new Uint32Array(nWords);
  for (let i = 0; i < nWords; i += 1) {
    let v = (pool[i % POOL_SIZE] ^ hasher.c) >>> 0;
    hasher.c = mul32(hasher.c, MULT_B);
    v = mul32(v, hasher.c);
    out[i] = (v ^ (v >>> XSHIFT)) >>> 0;
  }
  return out;
}

export class PCG64 {
  constructor(seed = 0) {
    // uint32 words viewed as uint64 little-endian: lo word first.
    const w = generateState(seedPool(seed), 8);
    const u64 = (i) => (BigInt(w[2 * i + 1]) << 32n) | BigInt(w[2 * i]);
    const initState = (u64(0) << 64n) | u64(1);
    const initSeq = (u64(2) << 64n) | u64(3);

    this.state = 0n;
    this.inc = ((initSeq << 1n) | 1n) & MASK128;
    this.step();
    this.state = (this.state + initState) & MASK128;
    this.step();

    this.hasUint32 = false;
    this.uinteger = 0;
  }

  step() {
    this.state = (this.state * PCG_MULT + this.inc) & MASK128;
  }

  /** XSL RR 128/64 output function, applied after a step. */
  nextUint64() {
    this.step();
    const s = this.state;
    const xored = ((s >> 64n) ^ s) & MASK64;
    const rot = Number((s >> 122n) & 63n);
    if (rot === 0) return xored;
    return ((xored >> BigInt(rot)) | (xored << BigInt(64 - rot))) & MASK64;
  }

  nextUint32() {
    if (this.hasUint32) {
      this.hasUint32 = false;
      return this.uinteger;
    }
    const next = this.nextUint64();
    this.hasUint32 = true;
    this.uinteger = Number((next >> 32n) & 0xffffffffn);
    return Number(next & 0xffffffffn);
  }

  /**
   * `integers(0, bound)` for `bound <= 2**32`, i.e. Lemire's bounded method with rejection.
   * Generator uses this; RandomState's masked rejection is a different sequence.
   */
  boundedUint32(rng) {
    const rngExcl = rng + 1;
    let m = BigInt(this.nextUint32()) * BigInt(rngExcl);
    let leftover = Number(m & 0xffffffffn);
    if (leftover < rngExcl) {
      const threshold = (U32 - rng) % rngExcl;
      while (leftover < threshold) {
        m = BigInt(this.nextUint32()) * BigInt(rngExcl);
        leftover = Number(m & 0xffffffffn);
      }
    }
    return Number(m >> 32n);
  }

  /** `rng.integers(0, high, size=count)` for `0 < high <= 2**32`. */
  integers(high, count) {
    const out = new Int32Array(count);
    const rng = high - 1;
    if (rng === 0) return out;
    if (rng > U32) throw new Error('bound beyond the 32-bit fast path is not implemented');
    for (let i = 0; i < count; i += 1) out[i] = this.boundedUint32(rng);
    return out;
  }
}
