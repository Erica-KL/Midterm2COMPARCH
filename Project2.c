#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── TIMING CONSTANTS (nanoseconds) ─────────────────────────────────────── */
#define HIT_NS   5      /* TLB hit: translation found in cache              */
#define MISS_NS  100    /* TLB miss: page-table walk required                */

/* ── SIMULATION PARAMETERS ──────────────────────────────────────────────── */
#define N_REFS    2000  /* memory references per run                         */
#define N_PAGES   256   /* virtual address space size in pages               */
#define N_SIZES   5     /* number of TLB sizes to test                       */
#define N_ALGOS   5     /* FIFO, LRU, LFU, CLOCK, OPTIMAL                   */
#define N_SCENS   5     /* scenarios                                         */
#define MAX_TLB   64    /* largest TLB size tested                           */

static const int TLB_SIZES[N_SIZES] = {4, 8, 16, 32, 64};

/* ── RESULT STORAGE ──────────────────────────────────────────────────────── */
typedef struct {
    double hit_rate;        /* percentage, 0–100                             */
    double avg_ns;          /* AMAT in nanoseconds                           */
    double lifespan_score;  /* normalized 0–100, higher = faster             */
    int    hits;
    int    misses;
} Result;

/* ── TLB STATE ───────────────────────────────────────────────────────────── */

/* Generic slot: holds a page number (-1 = empty) and metadata */
typedef struct {
    int page;       /* virtual page number currently cached  */
    int freq;       /* access count (used by LFU)            */
    int recency;    /* timestamp of last access (used by LRU)*/
    int use_bit;    /* second-chance bit (used by CLOCK)     */
} TLBSlot;

typedef struct {
    TLBSlot slots[MAX_TLB]; /* the actual TLB entries                       */
    int     size;           /* number of slots                              */
    int     count;          /* number of occupied slots                     */
    int     clock_hand;     /* CLOCK algorithm hand position                */
    int     fifo_head;      /* FIFO: index of oldest entry                  */
    int     timestamp;      /* global access counter (for LRU timestamps)   */
    int     hits;
    int     misses;
    long    total_ns;
} TLB;

/* Initialize TLB: clear all slots, reset counters */
static void tlb_init(TLB *t, int size) {
    memset(t, 0, sizeof(*t));
    t->size = size;
    for (int i = 0; i < size; i++)
        t->slots[i].page = -1; /* -1 means empty slot */
}

/* ── HELPER: find page in TLB, return index or -1 ───────────────────────── */
static int tlb_find(TLB *t, int page) {
    for (int i = 0; i < t->size; i++)
        if (t->slots[i].page == page)
            return i;
    return -1;
}

/* ── HELPER: find an empty slot ─────────────────────────────────────────── */
static int tlb_empty_slot(TLB *t) {
    for (int i = 0; i < t->size; i++)
        if (t->slots[i].page == -1)
            return i;
    return -1;
}

/* ── RECORD HIT or MISS ──────────────────────────────────────────────────── */
static void tlb_record(TLB *t, int hit) {
    if (hit) { t->hits++;   t->total_ns += HIT_NS;  }
    else     { t->misses++; t->total_ns += MISS_NS; }
}

/* ══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 1: FIFO  (First In First Out)
* ══════════════════════════════════════════════════════════════════════════ */
static void access_fifo(TLB *t, int page) {
    int idx = tlb_find(t, page);
    if (idx >= 0) {
        /* HIT: page already in TLB — nothing to update for FIFO */
        tlb_record(t, 1);
        return;
    }
    /* MISS: load the page */
    tlb_record(t, 0);
    int slot = tlb_empty_slot(t);
    if (slot < 0) {
        /* TLB full: evict the oldest entry (fifo_head position) */
        slot = t->fifo_head;
        t->fifo_head = (t->fifo_head + 1) % t->size; /* advance circular ptr */
    }
    t->slots[slot].page = page;
}

/* ══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 2: LRU  (Not Recently Used)
* ══════════════════════════════════════════════════════════════════════════ */
static void access_lru(TLB *t, int page) {
    t->timestamp++;
    int idx = tlb_find(t, page);
    if (idx >= 0) {
        /* HIT: update recency timestamp so this page stays "warm" */
        t->slots[idx].recency = t->timestamp;
        tlb_record(t, 1);
        return;
    }
    /* MISS */
    tlb_record(t, 0);
    int slot = tlb_empty_slot(t);
    if (slot < 0) {
        /* Find the least recently used slot: smallest recency value */
        int lru_time = t->slots[0].recency;
        slot = 0;
        for (int i = 1; i < t->size; i++) {
            if (t->slots[i].recency < lru_time) {
                lru_time = t->slots[i].recency;
                slot = i;
            }
        }
    }
    t->slots[slot].page    = page;
    t->slots[slot].recency = t->timestamp;
}

/* ══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 3: LFU  (Not Frequently Used)
* ══════════════════════════════════════════════════════════════════════════ */
static void access_lfu(TLB *t, int page) {
    int idx = tlb_find(t, page);
    if (idx >= 0) {
        /* HIT: increment frequency counter */
        t->slots[idx].freq++;
        tlb_record(t, 1);
        return;
    }
    /* MISS */
    tlb_record(t, 0);
    int slot = tlb_empty_slot(t);
    if (slot < 0) {
        /* Find slot with minimum frequency (least frequently used) */
        int min_freq = t->slots[0].freq;
        slot = 0;
        for (int i = 1; i < t->size; i++) {
            if (t->slots[i].freq < min_freq) {
                min_freq = t->slots[i].freq;
                slot = i;
            }
        }
    }
    t->slots[slot].page = page;
    t->slots[slot].freq = 1; /* start counter at 1 for the new load */
}

/* ══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 4: CLOCK (Second-Chance / NRU approximation)
 * ══════════════════════════════════════════════════════════════════════════ */
static void access_clock(TLB *t, int page) {
    int idx = tlb_find(t, page);
    if (idx >= 0) {
        /* HIT: set use bit — this page gets a "second chance" */
        t->slots[idx].use_bit = 1;
        tlb_record(t, 1);
        return;
    }
    /* MISS: sweep clock hand until we find a victim */
    tlb_record(t, 0);

    /* First check for empty slots (no eviction needed yet) */
    int slot = tlb_empty_slot(t);
    if (slot >= 0) {
        t->slots[slot].page    = page;
        t->slots[slot].use_bit = 1;
        return;
    }

    /* TLB full: run the CLOCK sweep */
    while (1) {
        if (t->slots[t->clock_hand].use_bit == 0) {
            /* Found a victim: use_bit is 0, this page gets no second chance */
            break;
        } else {
            /* Give second chance: clear the bit and advance */
            t->slots[t->clock_hand].use_bit = 0;
            t->clock_hand = (t->clock_hand + 1) % t->size;
        }
    }
    t->slots[t->clock_hand].page    = page;
    t->slots[t->clock_hand].use_bit = 1;
    t->clock_hand = (t->clock_hand + 1) % t->size; /* advance past new entry */
}

/* ══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 5: OPTIMAL (Belady's Algorithm)
 * ══════════════════════════════════════════════════════════════════════════ */
static void access_optimal(TLB *t, int page, const int *refs, int pos, int n) {
    int idx = tlb_find(t, page);
    if (idx >= 0) {
        tlb_record(t, 1);
        return;
    }
    tlb_record(t, 0);
    int slot = tlb_empty_slot(t);
    if (slot >= 0) {
        t->slots[slot].page = page;
        return;
    }

    /* Find next-use distance for each currently-cached page */
    int victim = 0;
    int max_dist = -1;
    for (int i = 0; i < t->size; i++) {
        int p = t->slots[i].page;
        /* Scan forward from pos+1 to find next occurrence of page p */
        int dist = n + 1; /* infinity: page never used again */
        for (int j = pos + 1; j < n; j++) {
            if (refs[j] == p) {
                dist = j - pos;
                break;
            }
        }
        if (dist > max_dist) {
            max_dist = dist;
            victim   = i; /* evict the page with the largest next-use distance */
        }
    }
    t->slots[victim].page = page;
}

/* ══════════════════════════════════════════════════════════════════════════
 * WORKLOAD GENERATORS
 *
 * Each generator fills refs[] with N_REFS page numbers that model a
 * particular memory access pattern.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Simple linear congruential RNG (no stdlib dependency) */
static unsigned int rng_state = 42;
static int rng_next(int max) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (int)(rng_state >> 16) % max;
}
static void rng_seed(unsigned int s) { rng_state = s; }

/*
 * SEQUENTIAL: pages 0,1,2,...,255,0,1,2,...
 * Models array scans, file reads. Zero temporal locality.
 * Every page is unique in a 256-step window → 0% hit rate for all
 * practical algorithms.
 */
static void gen_sequential(int *refs) {
    for (int i = 0; i < N_REFS; i++)
        refs[i] = i % N_PAGES;
}

/*
 * HOTSPOT: 85% of accesses to 8 "hot" pages, 15% random cold pages.
 * Models typical programs: tight inner loops + occasional data scatter.
 * LFU wins at small TLB sizes; all algorithms converge at size 16+.
 */
static void gen_hotspot(int *refs) {
    int hot[8];
    for (int i = 0; i < 8; i++) {
        hot[i] = rng_next(N_PAGES);
        /* ensure unique hot pages */
        for (int j = 0; j < i; j++)
            if (hot[j] == hot[i]) { i--; break; }
    }
    for (int i = 0; i < N_REFS; i++) {
        if (rng_next(100) < 85)
            refs[i] = hot[rng_next(8)];
        else
            refs[i] = rng_next(N_PAGES);
    }
}

/*
 * MULTI-LOCALITY: 4 regions of 16 pages each, center-biased (Gaussian-like).
 * Models programs with multiple hot loops competing for TLB space.
 * LFU leads because competing regions create frequency signals.
 */
static void gen_multi_locality(int *refs) {
    /* Define 4 regions at random offsets */
    int region_start[4];
    for (int r = 0; r < 4; r++)
        region_start[r] = rng_next(N_PAGES - 16);

    for (int i = 0; i < N_REFS; i++) {
        int r = rng_next(4);                        /* pick a random region  */
        /* Center-bias within region: try 3 times and take the middle value  */
        /* (a simple approximation of Gaussian center-weighting)             */
        int a = rng_next(16), b = rng_next(16), c = rng_next(16);
        int offset = a < b ? (b < c ? b : (a < c ? c : a))
                           : (a < c ? a : (b < c ? c : b));
        refs[i] = (region_start[r] + offset) % N_PAGES;
    }
}

/*
 * RANDOM: uniform random across all pages.
 * Adversarial case. No locality → all practical algorithms perform identically.
 * Only OPTIMAL exploits the small collision probability in random sequences.
 */
static void gen_random(int *refs) {
    for (int i = 0; i < N_REFS; i++)
        refs[i] = rng_next(N_PAGES);
}

/*
 * TEMPORAL DECAY: a 12-page working set that slowly rotates.
 * With 0.3% probability per access, one hot page is replaced by a new one.
 * Models phase shifts: startup → steady state → cleanup.
 * At TLB size >= working-set size (12 pages), all algorithms saturate near 100%.
 */
static void gen_temporal_decay(int *refs) {
    int active[12];
    for (int i = 0; i < 12; i++) {
        active[i] = rng_next(N_PAGES);
        for (int j = 0; j < i; j++)
            if (active[j] == active[i]) { i--; break; }
    }
    for (int i = 0; i < N_REFS; i++) {
        /* Occasionally rotate a page out of the active set */
        if (rng_next(1000) < 3) {        /* ~0.3% chance per access */
            int out = rng_next(12);
            int np;
            do { np = rng_next(N_PAGES); } while (np == active[out]);
            active[out] = np;
        }
        refs[i] = active[rng_next(12)];
    }
}

/* ── RESULT COMPUTATION ──────────────────────────────────────────────────── */
static Result compute_result(TLB *t) {
    Result r;
    int total = t->hits + t->misses;
    r.hits     = t->hits;
    r.misses   = t->misses;
    r.hit_rate = total ? (double)t->hits / total * 100.0 : 0.0;
    r.avg_ns   = total ? (double)t->total_ns / total : MISS_NS;
    /* Lifespan score: normalized efficiency vs worst case (all misses) */
    r.lifespan_score = (MISS_NS - r.avg_ns) / (double)(MISS_NS - HIT_NS) * 100.0;
    if (r.lifespan_score < 0)   r.lifespan_score = 0;
    if (r.lifespan_score > 100) r.lifespan_score = 100;
    return r;
}

/* ── SCENARIO RUNNER ─────────────────────────────────────────────────────── */
typedef struct {
    const char *name;
    void (*gen)(int *refs);
} Scenario;

static void run_scenario(const char *name, int *refs, int tlb_size) {
    Result results[N_ALGOS];
    const char *names[N_ALGOS] = {"FIFO","LRU","LFU","CLOCK","OPTIMAL"};

    /* ── FIFO ── */
    { TLB t; tlb_init(&t, tlb_size);
      for (int i = 0; i < N_REFS; i++) access_fifo(&t, refs[i]);
      results[0] = compute_result(&t); }

    /* ── LRU ── */
    { TLB t; tlb_init(&t, tlb_size);
      for (int i = 0; i < N_REFS; i++) access_lru(&t, refs[i]);
      results[1] = compute_result(&t); }

    /* ── LFU ── */
    { TLB t; tlb_init(&t, tlb_size);
      for (int i = 0; i < N_REFS; i++) access_lfu(&t, refs[i]);
      results[2] = compute_result(&t); }

    /* ── CLOCK ── */
    { TLB t; tlb_init(&t, tlb_size);
      for (int i = 0; i < N_REFS; i++) access_clock(&t, refs[i]);
      results[3] = compute_result(&t); }

    /* ── OPTIMAL ── */
    { TLB t; tlb_init(&t, tlb_size);
      for (int i = 0; i < N_REFS; i++)
          access_optimal(&t, refs[i], refs, i, N_REFS);
      results[4] = compute_result(&t); }

    /* Print table row */
    printf("  %-12s | size=%2d |", name, tlb_size);
    for (int a = 0; a < N_ALGOS; a++)
        printf(" %-7s hit=%5.1f%% avg=%5.1fns |",
               names[a], results[a].hit_rate, results[a].avg_ns);
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN: run all scenarios × all TLB sizes
 * ══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    int refs[N_REFS];

    Scenario scenarios[N_SCENS] = {
        { "Sequential",     gen_sequential     },
        { "Hotspot",        gen_hotspot        },
        { "Multi-Locality", gen_multi_locality },
        { "Random",         gen_random         },
        { "Temp.Decay",     gen_temporal_decay },
    };

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         TLB Replacement Algorithm Benchmark (C)             ║\n");
    printf("║  Hit cost: %d ns    Miss cost: %d ns    Refs: %d             ║\n",
           HIT_NS, MISS_NS, N_REFS);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    for (int s = 0; s < N_SCENS; s++) {
        printf("━━━ Scenario: %-20s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
               scenarios[s].name);

        for (int si = 0; si < N_SIZES; si++) {
            rng_seed(42);                    /* deterministic per scenario   */
            scenarios[s].gen(refs);          /* generate reference string    */
            run_scenario(scenarios[s].name, refs, TLB_SIZES[si]);
        }
        printf("\n");
    }

    /* ── AMAT demonstration (from the AMAT formula source) ── */
    printf("━━━ AMAT Formula Demonstration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  AMAT = hit_time + miss_rate * miss_penalty\n");
    printf("       = %d ns + miss_rate * %d ns\n\n", HIT_NS, MISS_NS);

    double example_hit_rates[] = {0.0, 0.25, 0.50, 0.75, 0.85, 0.95, 0.99};
    printf("  %-12s  %-15s  %-15s\n", "Hit Rate", "Miss Rate", "AMAT (ns)");
    printf("  %-12s  %-15s  %-15s\n",
           "------------", "---------------", "---------------");
    for (int i = 0; i < 7; i++) {
        double hr = example_hit_rates[i];
        double mr = 1.0 - hr;
        double amat = HIT_NS + mr * MISS_NS;
        printf("  %10.0f%%  %13.0f%%  %13.2f ns\n", hr*100, mr*100, amat);
    }
    printf("\n  → Going from 75%% to 99%% hit rate cuts AMAT from %.1f to %.1f ns (%.1fx speedup)\n\n",
           HIT_NS + 0.25*MISS_NS, HIT_NS + 0.01*MISS_NS,
           (HIT_NS + 0.25*MISS_NS) / (HIT_NS + 0.01*MISS_NS));

    return 0;
}