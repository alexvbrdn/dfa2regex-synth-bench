//! The benchmark's DFA ids: a key of a minimal DFA that does not depend on how its states are
//! numbered or its bases ordered. This is a copy of the code of the generator that made the
//! benchmark.
//!
//! The automaton is determinized and minimized, and its alphabet split into *bases*: the
//! coarsest disjoint character classes its transitions are made of (RegexSolver's minimal
//! spanning set). The characters on no transition are not a base, so `a*` and `.*` have the
//! same id: one base, looping on the start state. The DFA is then written down for an order
//! of its bases, its states numbered in breadth-first order from the start (state 0), and the
//! smallest of these encodings over all base orders is kept. Only the orders that can matter
//! are tried: bases are first told apart by what their transitions do (color refinement:
//! repeatedly grouping bases, and states, by the groups of what they lead to); only bases left
//! in the same group are *tied* and permuted. The id is the `xxh3_128` of the encoding, its 16
//! bytes (little endian) in hex.
//!
//! When more than [`MAX_BASE_PERMUTATIONS`] orders of tied bases would have to be compared,
//! the ties are broken by the order RegexSolver gives the bases, and the id is not guaranteed
//! canonical (`exact: false`): two DFAs of the same shape could get different ids. This
//! happens for about 1 DFA in 2 million from the benchmark's generators.

use regexsolver::fast_automaton::FastAutomaton;
use xxhash_rust::xxh3::xxh3_128;

use crate::data::Error;

pub const MAX_BASE_PERMUTATIONS: usize = 5040;

const DEAD: u32 = u32::MAX;

/// A DFA's id and what it was computed from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    /// The benchmark's `id`: 32 hex digits.
    pub id: String,
    /// States and bases of the minimal DFA.
    pub states: usize,
    pub bases: usize,
    /// Whether every tied base order was enumerated (the key is canonical).
    pub exact: bool,
}

/// The minimal DFA of `automaton`, with its minimal spanning set: the coarsest one its
/// transitions allow, so the bases depend on the language only, and no transition uses the rest.
pub fn minimal_dfa(automaton: &FastAutomaton) -> Result<FastAutomaton, Error> {
    let mut dfa = automaton.determinize()?.into_owned();
    dfa.minimize()?;
    dfa.recompute_minimal_spanning_set()?;
    Ok(dfa)
}

/// The id of the language of `automaton` (any automaton: it is minimized first).
pub fn key(automaton: &FastAutomaton) -> Result<Key, Error> {
    canonicalize(&minimal_dfa(automaton)?)
}

/// The id of a minimal DFA whose spanning set is minimal (see [`minimal_dfa`]).
pub fn canonicalize(automaton: &FastAutomaton) -> Result<Key, Error> {
    if !automaton.is_deterministic() {
        return Err("canonicalize requires a DFA".into());
    }
    if !automaton.is_minimal() {
        return Err("canonicalize requires a minimal DFA".into());
    }
    let spanning_set = automaton.spanning_set();
    let k = spanning_set.number_of_spanning_ranges();
    // RegexSolver's condition bits put the rest first when it is non-empty: base b is bit
    // b + offset.
    let offset = usize::from(!spanning_set.rest().is_empty());
    if k > 64 {
        return Err(format!("{k} bases exceed the 64-bit mask").into());
    }

    let raw_states = automaton.states_vec();
    let n = raw_states.len();
    if n > u16::MAX as usize {
        return Err(format!("{n} states exceed u16").into());
    }
    let max_raw = raw_states.iter().max().map_or(0, |s| s + 1);
    let mut dense = vec![u32::MAX; max_raw];
    for (i, s) in raw_states.iter().enumerate() {
        dense[*s] = i as u32;
    }

    let mut delta = vec![DEAD; n * k];
    for (i, &from) in raw_states.iter().enumerate() {
        for (condition, &to) in automaton.transitions_from(from) {
            for (bit, set) in condition.binary_representation().into_iter().enumerate() {
                if set {
                    let Some(b) = bit.checked_sub(offset) else {
                        return Err(
                            "a transition uses the rest: the spanning set is not minimal".into(),
                        );
                    };
                    let cell = &mut delta[i * k + b];
                    if *cell != DEAD {
                        return Err("overlapping transitions".into());
                    }
                    *cell = dense[to];
                }
            }
        }
    }
    let start = dense[automaton.start_state()] as usize;
    let accept: Vec<bool> = raw_states
        .iter()
        .map(|s| automaton.is_accepted(*s))
        .collect();

    let table = Table {
        n,
        k,
        delta,
        start,
        accept,
    };
    let (best, exact) = table.canonical_encoding();
    let hash = xxh3_128(&best.encode(n, k));
    Ok(Key {
        id: hash
            .to_le_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        states: n,
        bases: k,
        exact,
    })
}

struct Table {
    n: usize,
    k: usize,
    /// `delta[s * k + b]`: target of state `s` on base `b`, or [`DEAD`].
    delta: Vec<u32>,
    start: usize,
    accept: Vec<bool>,
}

/// The DFA renumbered for one base order.
struct Relabeled {
    accepts: Vec<u32>,
    /// Canonical delta, row-major over canonical states and bases.
    delta: Vec<u32>,
}

impl Relabeled {
    fn cmp_key(&self) -> (&[u32], &[u32]) {
        (&self.accepts, &self.delta)
    }

    fn encode(&self, n: usize, k: usize) -> Vec<u8> {
        let mut key = Vec::with_capacity(6 + 2 * (self.accepts.len() + self.delta.len()));
        key.extend_from_slice(&(n as u16).to_le_bytes());
        key.extend_from_slice(&(k as u16).to_le_bytes());
        key.extend_from_slice(&(self.accepts.len() as u16).to_le_bytes());
        for a in &self.accepts {
            key.extend_from_slice(&(*a as u16).to_le_bytes());
        }
        // 0 = dead, t + 1 otherwise.
        for t in &self.delta {
            let v = if *t == DEAD { 0 } else { *t as u16 + 1 };
            key.extend_from_slice(&v.to_le_bytes());
        }
        key
    }
}

impl Table {
    fn target(&self, s: usize, b: usize) -> Option<usize> {
        let t = self.delta[s * self.k + b];
        (t != DEAD).then_some(t as usize)
    }

    /// The smallest encoding, and whether all tied orders were enumerated.
    fn canonical_encoding(&self) -> (Relabeled, bool) {
        let base_colors = self.refine();

        let mut order: Vec<usize> = (0..self.k).collect();
        order.sort_by_key(|b| (base_colors[*b], *b));
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut i = 0;
        while i < self.k {
            let mut j = i + 1;
            while j < self.k && base_colors[order[j]] == base_colors[order[i]] {
                j += 1;
            }
            if j - i > 1 {
                groups.push(i..j);
            }
            i = j;
        }

        let mut product: usize = 1;
        let exact = groups.iter().all(|g| {
            product = (1..=g.len()).fold(product, |p, f| p.saturating_mul(f));
            product <= MAX_BASE_PERMUTATIONS
        });
        if !exact {
            return (self.relabel(&order), false);
        }

        let mut best = self.relabel(&order);
        // Odometer over per-group permutations; each group starts sorted (ascending
        // original index) and `next_permutation` wraps it back.
        loop {
            let mut advanced = false;
            for g in &groups {
                if next_permutation(&mut order[g.clone()]) {
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                break;
            }
            let candidate = self.relabel(&order);
            if candidate.cmp_key() < best.cmp_key() {
                best = candidate;
            }
        }
        (best, true)
    }

    fn relabel(&self, order: &[usize]) -> Relabeled {
        let (n, k) = (self.n, self.k);
        let mut numbering = vec![DEAD; n];
        let mut queue = Vec::with_capacity(n);
        numbering[self.start] = 0;
        queue.push(self.start);
        let mut head = 0;
        while head < queue.len() {
            let s = queue[head];
            head += 1;
            for &b in order {
                if let Some(t) = self.target(s, b)
                    && numbering[t] == DEAD
                {
                    numbering[t] = queue.len() as u32;
                    queue.push(t);
                }
            }
        }
        // A minimal DFA has every state reachable; number any stragglers anyway so the
        // encoding stays total.
        for (s, number) in numbering.iter_mut().enumerate() {
            if *number == DEAD {
                *number = queue.len() as u32;
                queue.push(s);
            }
        }

        let mut delta = vec![DEAD; n * k];
        for (new_s, &s) in queue.iter().enumerate() {
            for (j, &b) in order.iter().enumerate() {
                if let Some(t) = self.target(s, b) {
                    delta[new_s * k + j] = numbering[t];
                }
            }
        }
        let mut accepts: Vec<u32> = (0..n)
            .filter(|s| self.accept[*s])
            .map(|s| numbering[s])
            .collect();
        accepts.sort_unstable();
        Relabeled { accepts, delta }
    }

    /// Alternating color refinement of states and bases; returns the stable base colors.
    /// Colors are ranks of sorted signatures, so they are invariant under renumbering and
    /// relabeling.
    fn refine(&self) -> Vec<u32> {
        let (n, k) = (self.n, self.k);
        let mut state_colors = ranks(
            &(0..n)
                .map(|s| {
                    let out = (0..k).filter(|b| self.target(s, *b).is_some()).count();
                    vec![(s == self.start) as u32, self.accept[s] as u32, out as u32]
                })
                .collect::<Vec<_>>(),
        );
        let mut base_colors = ranks(
            &(0..k)
                .map(|b| {
                    let (mut defined, mut loops, mut into_accept) = (0, 0, 0);
                    for s in 0..n {
                        if let Some(t) = self.target(s, b) {
                            defined += 1;
                            loops += (t == s) as u32;
                            into_accept += self.accept[t] as u32;
                        }
                    }
                    vec![defined, loops, into_accept]
                })
                .collect::<Vec<_>>(),
        );

        let classes = |colors: &[u32]| colors.iter().max().map_or(0, |m| m + 1);
        let mut total = classes(&state_colors) + classes(&base_colors);
        loop {
            let color_of = |t: Option<usize>, colors: &[u32]| t.map_or(DEAD, |t| colors[t]);
            state_colors = ranks(
                &(0..n)
                    .map(|s| {
                        let mut sig: Vec<(u32, u32)> = (0..k)
                            .map(|b| (base_colors[b], color_of(self.target(s, b), &state_colors)))
                            .collect();
                        sig.sort_unstable();
                        std::iter::once(state_colors[s])
                            .chain(sig.into_iter().flat_map(|(a, b)| [a, b]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            base_colors = ranks(
                &(0..k)
                    .map(|b| {
                        let mut sig: Vec<(u32, u32)> = (0..n)
                            .map(|s| (state_colors[s], color_of(self.target(s, b), &state_colors)))
                            .collect();
                        sig.sort_unstable();
                        std::iter::once(base_colors[b])
                            .chain(sig.into_iter().flat_map(|(a, b)| [a, b]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
            let new_total = classes(&state_colors) + classes(&base_colors);
            if new_total == total {
                return base_colors;
            }
            total = new_total;
        }
    }
}

/// Maps each signature to its rank among the distinct sorted signatures.
fn ranks<T: Ord + Clone>(signatures: &[T]) -> Vec<u32> {
    let mut sorted = signatures.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    signatures
        .iter()
        .map(|s| sorted.binary_search(s).unwrap() as u32)
        .collect()
}

/// Advances `v` to its next lexicographic permutation; on the last one, resets it to ascending
/// order and returns `false`.
fn next_permutation(v: &mut [usize]) -> bool {
    if v.len() < 2 {
        return false;
    }
    let mut i = v.len() - 1;
    while i > 0 && v[i - 1] >= v[i] {
        i -= 1;
    }
    if i == 0 {
        v.reverse();
        return false;
    }
    let mut j = v.len() - 1;
    while v[j] <= v[i - 1] {
        j -= 1;
    }
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use regexsolver::{
        CharRange, fast_automaton::FastAutomaton, regex::RegularExpression,
        regex_charclass::char::Char,
    };

    use super::*;

    fn id(pattern: &str) -> String {
        key(&RegularExpression::new(pattern)
            .unwrap()
            .to_automaton()
            .unwrap())
        .unwrap()
        .id
    }

    fn range(c: char) -> CharRange {
        CharRange::new_from_range(Char::new(c)..=Char::new(c))
    }

    #[test]
    fn same_language_different_regexes() {
        assert_eq!(id("(ab|cd)+e"), id("(cd|ab)(ab|cd)*e"));
        assert_eq!(id("a(ba)*"), id("(ab)*a"));
    }

    #[test]
    fn bases_covering_every_character() {
        // A base may hold every character but a few: [^a] is a base like b.
        assert_eq!(id("[^a]"), id("b"));
        assert_eq!(id("a[^a]*"), id("ab*"));
        assert_eq!(id(".*"), id("a*"));
        // The same from the benchmark's format, where the bases then cover every character.
        let bases = ["a".to_string(), "[^a]".to_string()];
        let dfa = crate::data::automaton(2, &[1], &bases, &[(0, 1, vec![0]), (1, 1, vec![1])]);
        assert_eq!(key(&dfa.unwrap()).unwrap().id, id("ab*"));
    }

    #[test]
    fn base_relabeling() {
        assert_eq!(id("a*b"), id("b*a"));
        assert_eq!(id("x(y|z)*w"), id("q(r|s)*t"));
        assert_eq!(id("a*"), id("[^a]*"));
        assert_ne!(id("ab"), id("ab*"));
        assert_ne!(id("a*b"), id("a*a"));
    }

    #[test]
    fn alphabet_coverage_is_not_a_base() {
        assert_eq!(id("a*"), id(".*"));
        assert_eq!(id("ab*"), id("a[^a]*"));
        assert_eq!(id("(a|bc)*"), id("([^bc]|bc)*"));
    }

    #[test]
    fn state_renumbering() {
        // The DFA of `a(b|c)*d`, built with shuffled state ids.
        let mut automaton = FastAutomaton::new_empty();
        let s = [
            automaton.new_state(),
            automaton.new_state(),
            automaton.new_state(),
        ];
        let (s_end, s_loop) = (s[0], s[2]);
        automaton
            .add_transition_from_range(0, s_loop, &range('a'))
            .unwrap();
        automaton
            .add_transition_from_range(s_loop, s_loop, &range('b').union(&range('c')))
            .unwrap();
        automaton
            .add_transition_from_range(s_loop, s_end, &range('d'))
            .unwrap();
        automaton.accept(s_end);
        automaton.remove_state(s[1]);
        assert_eq!(key(&automaton).unwrap().id, id("a(b|c)*d"));
    }

    #[test]
    fn next_permutation_enumerates_all() {
        let mut v = vec![0, 1, 2, 3];
        let mut count = 1;
        while next_permutation(&mut v) {
            count += 1;
        }
        assert_eq!(count, 24);
        assert_eq!(v, vec![0, 1, 2, 3]);
    }
}
