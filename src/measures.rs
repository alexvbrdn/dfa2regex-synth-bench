//! The measures of a regular expression, computed from its text:
//!
//! - `complexity`: RegexSolver's `evaluate_complexity` (version 1.1.0) of the pattern as
//!   RegexSolver reads it (`RegularExpression::parse(text, false)`, which keeps its structure
//!   but for a few rewrites: a run of one character becomes a counted repetition, `bb` as
//!   `b{2}`; an alternation whose branches are all single literal characters one class, `b|c`
//!   as `[bc]`; groups and `{1}` disappear, `{0,1}` is `?` and `{m,m}` is `{m}`), the primary
//!   measure;
//! - `length`: the number of characters (Unicode scalar values) of the text;
//! - `alphabetic_width`: occurrences of single-character atoms: literals, classes (`[a-c]`,
//!   `\d`) and `.`;
//! - `tree_size`: the nodes of the syntax tree with binary operators: an atom is 1, a
//!   concatenation or an alternation of n >= 2 parts is its parts plus n - 1, an empty
//!   expression is 1, a quantifier (`*`, `+`, `?`, `{m}`, `{m,n}`, `{m,}`) is its operand plus 1,
//!   a group nothing;
//! - `star_height`: the deepest nesting of unbounded quantifiers (`*`, `+`, `{m,}`).
//!
//! Complexity, length, alphabetic width and tree size are sizes; star height is a depth. The
//! last four are the standard descriptional-complexity measures (Gruber and Holzer, "From
//! finite automata to regular expressions and back -- a summary on descriptional complexity",
//! 2015), on the syntax tree of the text as written (`regex-syntax`'s `ast`: `b|c` is an
//! alternation of two atoms, a group adds nothing). RegexSolver's parser is not used for them:
//! it rewrites some forms (`b|c` into `[bc]`).

use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange, parse::ParserBuilder};
use regexsolver::regex::RegularExpression;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Measures {
    pub complexity: f64,
    pub length: usize,
    pub alphabetic_width: usize,
    pub tree_size: usize,
    pub star_height: usize,
}

/// RegexSolver's reading of the pattern, or `None` if it does not parse.
pub fn parse(text: &str) -> Option<RegularExpression> {
    RegularExpression::parse(text, false).ok()
}

/// The measures of `text`, with `regex` its RegexSolver parse; `None` if `regex-syntax` cannot
/// read it (a pattern outside the benchmark's syntax).
pub fn measures(text: &str, regex: &RegularExpression) -> Option<Measures> {
    let (alphabetic_width, tree_size, star_height) = syntax_measures(text)?;
    Some(Measures {
        complexity: regex.evaluate_complexity(),
        length: text.chars().count(),
        alphabetic_width,
        tree_size,
        star_height,
    })
}

/// `(alphabetic width, tree size, star height)` of the text as written. Iterative, and without
/// a nesting limit: state elimination's expressions nest deeply.
pub fn syntax_measures(text: &str) -> Option<(usize, usize, usize)> {
    let ast = ParserBuilder::new()
        .nest_limit(u32::MAX)
        .build()
        .parse(text)
        .ok()?;
    let (mut width, mut size, mut star_height) = (0, 0, 0);
    let mut stack = vec![(&ast, 0usize)];
    while let Some((node, stars)) = stack.pop() {
        star_height = star_height.max(stars);
        match node {
            Ast::Literal(_)
            | Ast::Dot(_)
            | Ast::ClassBracketed(_)
            | Ast::ClassPerl(_)
            | Ast::ClassUnicode(_) => {
                width += 1;
                size += 1;
            }
            Ast::Empty(_) => size += 1,
            Ast::Group(g) => stack.push((&g.ast, stars)),
            Ast::Concat(c) => {
                size += c.asts.len().saturating_sub(1);
                stack.extend(c.asts.iter().map(|a| (a, stars)));
            }
            Ast::Alternation(a) => {
                size += a.asts.len().saturating_sub(1);
                stack.extend(a.asts.iter().map(|x| (x, stars)));
            }
            Ast::Repetition(r) => {
                size += 1;
                let unbounded = matches!(
                    r.op.kind,
                    RepetitionKind::ZeroOrMore
                        | RepetitionKind::OneOrMore
                        | RepetitionKind::Range(RepetitionRange::AtLeast(_))
                );
                let stars = stars + usize::from(unbounded);
                star_height = star_height.max(stars);
                stack.push((&r.ast, stars));
            }
            // Anchors and flags: not part of the benchmark's syntax.
            Ast::Assertion(_) | Ast::Flags(_) => return None,
        }
    }
    Some((width, size, star_height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(text: &str) -> Measures {
        measures(text, &parse(text).unwrap()).unwrap()
    }

    #[test]
    fn standard_measures() {
        // (ab|c)*d: classes a, b, c, d; a concatenation of 2, an alternation of 2, a star and a
        // concatenation of 2: 4 + 1 + 1 + 1 + 1 = 8 nodes; one star.
        let m = of("(ab|c)*d");
        assert_eq!(
            (m.length, m.alphabetic_width, m.tree_size, m.star_height),
            (8, 4, 8, 1)
        );
        assert_eq!(of("(ab|ca)*(bd)").alphabetic_width, 6);
        assert_eq!(of("((a*b)*c)*").star_height, 3);
        let m = of("a{2,5}b?");
        assert_eq!((m.tree_size, m.star_height), (5, 0));
        assert_eq!(of("a{3,}").star_height, 1);
        assert_eq!(of(".*").alphabetic_width, 1);
        // RegexSolver reads bb as b{2}: the same score for both texts
        assert_eq!(of("abb").complexity, of("ab{2}").complexity);
        // as written: b|c is two classes and an alternation, [bc] one class
        assert_eq!((of("b|c").alphabetic_width, of("b|c").tree_size), (2, 3));
        assert_eq!((of("[bc]").alphabetic_width, of("[bc]").tree_size), (1, 1));
        // RegexSolver's score is the same for both: it reads b|c as [bc]
        assert_eq!(of("b|c").complexity, of("[bc]").complexity);
    }

    #[test]
    fn deep_nesting() {
        let text = format!("{}a{}", "(".repeat(5000), ")*".repeat(5000));
        assert_eq!(syntax_measures(&text), Some((1, 5001, 5000)));
    }
}
