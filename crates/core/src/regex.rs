//! The accepted regex subset (`contracts/specs/schema.md`), for `Schema.pattern` and
//! the guard DSL's `matches`.
//!
//! Implicitly anchored at both ends; literals, classes `[...]` with ranges and
//! negation, `.`, the escapes `\d \w \s \D \W \S`, non-capturing grouping `(...)`,
//! alternation `|`, quantifiers `? * + {m} {m,n}` with `n ≤ 256`; nesting depth ≤ 8,
//! pattern length ≤ 512. No backreferences, lookaround, lazy quantifiers or flags.
//!
//! Everything compiles to a Thompson NFA simulated breadth-first, so evaluation is
//! linear in the input and backtracking cannot occur. A pattern outside the subset is
//! a validation error, not a silently-degraded match.

use agent_contracts::Diagnostic;

const MAX_PATTERN_LEN: usize = 512;
const MAX_NESTING: usize = 8;
const MAX_REPEAT: u32 = 256;
/// v1 compile limit: bounds the `{m,n}`-expansion blowup that the syntactic limits
/// alone do not (e.g. nested `{256}`).
const MAX_NFA_STATES: usize = 10_000;

/// A compiled pattern. Equality is structural over the parsed AST — the
/// "canonicalised pattern" comparison `Subsumes` uses (v1 rule: literal equality).
#[derive(Debug, Clone)]
pub struct Regex {
    source: String,
    ast: Ast,
    nfa: Nfa,
}

impl PartialEq for Regex {
    fn eq(&self, other: &Self) -> bool {
        self.ast == other.ast
    }
}
impl Eq for Regex {}

impl Regex {
    pub fn parse(pattern: &str) -> Result<Regex, Diagnostic> {
        Self::parse_at(pattern, "$")
    }

    /// `path` names the field holding the pattern, for diagnostics.
    pub fn parse_at(pattern: &str, path: &str) -> Result<Regex, Diagnostic> {
        if pattern.chars().count() > MAX_PATTERN_LEN {
            return Err(err(
                path,
                format!("pattern longer than {MAX_PATTERN_LEN} characters"),
            ));
        }
        let ast = Parser {
            chars: pattern.chars().collect(),
            pos: 0,
            path,
        }
        .parse()?;
        let nfa = Nfa::compile(&ast).map_err(|m| err(path, m))?;
        Ok(Regex {
            source: pattern.to_owned(),
            ast,
            nfa,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Full-string match (both ends anchored).
    pub fn is_match(&self, input: &str) -> bool {
        self.nfa.is_match(input)
    }
}

fn err(path: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error("invalid_pattern", path, message)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Ast {
    Empty,
    Literal(char),
    /// Any char.
    Dot,
    Class {
        negated: bool,
        items: Vec<ClassItem>,
    },
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat {
        node: Box<Ast>,
        min: u32,
        max: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ClassItem {
    Single(char),
    Range(char, char),
    Perl(Perl),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Perl {
    Digit,
    Word,
    Space,
    NotDigit,
    NotWord,
    NotSpace,
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    path: &'a str,
}

impl Parser<'_> {
    fn parse(mut self) -> Result<Ast, Diagnostic> {
        let ast = self.alternation(0)?;
        if self.pos < self.chars.len() {
            return Err(self.err(format!("unexpected {:?}", self.chars[self.pos])));
        }
        Ok(ast)
    }

    fn err(&self, message: impl Into<String>) -> Diagnostic {
        let m = message.into();
        err(self.path, format!("at offset {}: {m}", self.pos))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn alternation(&mut self, depth: usize) -> Result<Ast, Diagnostic> {
        let mut branches = vec![self.concat(depth)?];
        while self.peek() == Some('|') {
            self.bump();
            branches.push(self.concat(depth)?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().expect("one branch")
        } else {
            Ast::Alt(branches)
        })
    }

    fn concat(&mut self, depth: usize) -> Result<Ast, Diagnostic> {
        let mut parts = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            parts.push(self.repeat(depth)?);
        }
        Ok(match parts.len() {
            0 => Ast::Empty,
            1 => parts.pop().expect("one part"),
            _ => Ast::Concat(parts),
        })
    }

    fn repeat(&mut self, depth: usize) -> Result<Ast, Diagnostic> {
        let atom = self.atom(depth)?;
        let (min, max) = match self.peek() {
            Some('?') => {
                self.bump();
                (0, Some(1))
            }
            Some('*') => {
                self.bump();
                (0, None)
            }
            Some('+') => {
                self.bump();
                (1, None)
            }
            Some('{') => {
                self.bump();
                self.counted()?
            }
            _ => return Ok(atom),
        };
        if matches!(self.peek(), Some('?')) {
            return Err(self.err("lazy quantifiers are not in the subset"));
        }
        if matches!(self.peek(), Some('?' | '*' | '+' | '{')) {
            return Err(self.err("a quantifier may not be quantified"));
        }
        if matches!(atom, Ast::Empty) {
            return Err(self.err("nothing to repeat"));
        }
        Ok(Ast::Repeat {
            node: Box::new(atom),
            min,
            max,
        })
    }

    fn counted(&mut self) -> Result<(u32, Option<u32>), Diagnostic> {
        let m = self.number()?;
        let n = match self.bump() {
            Some('}') => m,
            Some(',') => {
                let n = self.number()?;
                match self.bump() {
                    Some('}') => n,
                    _ => return Err(self.err("expected '}' after {m,n}")),
                }
            }
            _ => return Err(self.err("expected '}' or ',' in a counted repetition ({m} or {m,n}; open-ended {m,} is not in the subset)")),
        };
        if n > MAX_REPEAT {
            return Err(self.err(format!("repetition bound {n} exceeds {MAX_REPEAT}")));
        }
        if m > n {
            return Err(self.err(format!("repetition {{{m},{n}}} has m > n")));
        }
        Ok((m, Some(n)))
    }

    fn number(&mut self) -> Result<u32, Diagnostic> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.pos == start {
            return Err(self.err("expected a number"));
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse()
            .map_err(|_| self.err("repetition bound out of range"))
    }

    fn atom(&mut self, depth: usize) -> Result<Ast, Diagnostic> {
        match self.bump() {
            Some('(') => {
                if self.peek() == Some('?') {
                    return Err(self.err("group modifiers '(?' are not in the subset; groups are plain '(...)' and non-capturing"));
                }
                if depth + 1 > MAX_NESTING {
                    return Err(self.err(format!("nesting deeper than {MAX_NESTING}")));
                }
                let inner = self.alternation(depth + 1)?;
                match self.bump() {
                    Some(')') => Ok(inner),
                    _ => Err(self.err("unclosed group")),
                }
            }
            Some('[') => {
                if depth + 1 > MAX_NESTING {
                    return Err(self.err(format!("nesting deeper than {MAX_NESTING}")));
                }
                self.class()
            }
            Some('.') => Ok(Ast::Dot),
            Some('\\') => Ok(self.escape(false)?),
            Some(c @ ('^' | '$')) => Err(self.err(format!(
                "{c:?} is not in the subset: patterns are implicitly anchored at both ends"
            ))),
            Some(c @ ('*' | '+' | '?' | '{')) => {
                Err(self.err(format!("quantifier {c:?} with nothing to repeat")))
            }
            Some(')') | None => Err(self.err("expected an atom")),
            Some(c) => Ok(Ast::Literal(c)),
        }
    }

    fn escape(&mut self, in_class: bool) -> Result<Ast, Diagnostic> {
        let c = self.bump().ok_or_else(|| self.err("dangling '\\'"))?;
        let perl = match c {
            'd' => Some(Perl::Digit),
            'w' => Some(Perl::Word),
            's' => Some(Perl::Space),
            'D' => Some(Perl::NotDigit),
            'W' => Some(Perl::NotWord),
            'S' => Some(Perl::NotSpace),
            _ => None,
        };
        if let Some(p) = perl {
            return Ok(Ast::Class {
                negated: false,
                items: vec![ClassItem::Perl(p)],
            });
        }
        match c {
            'n' => Ok(Ast::Literal('\n')),
            't' => Ok(Ast::Literal('\t')),
            'r' => Ok(Ast::Literal('\r')),
            c if !c.is_alphanumeric() => Ok(Ast::Literal(c)),
            _ => {
                let where_ = if in_class { "in a class" } else { "" };
                Err(self.err(format!("unsupported escape '\\{c}' {where_}")))
            }
        }
    }

    fn class(&mut self) -> Result<Ast, Diagnostic> {
        let negated = if self.peek() == Some('^') {
            self.bump();
            true
        } else {
            false
        };
        let mut items = Vec::new();
        loop {
            let c = match self.bump() {
                None => return Err(self.err("unclosed character class")),
                Some(']') if !items.is_empty() || negated => break,
                Some(']') => break, // "[]" = empty class; caught below
                Some(c) => c,
            };
            let lo = if c == '\\' {
                match self.escape(true)? {
                    Ast::Literal(l) => Some(l),
                    Ast::Class { items: perl, .. } => {
                        items.extend(perl);
                        None
                    }
                    _ => unreachable!("escape yields literal or class"),
                }
            } else {
                Some(c)
            };
            let Some(lo) = lo else { continue };
            if self.peek() == Some('-')
                && self.chars.get(self.pos + 1).copied() != Some(']')
                && self.chars.get(self.pos + 1).is_some()
            {
                self.bump(); // '-'
                let hi = match self.bump().expect("checked above") {
                    '\\' => match self.escape(true)? {
                        Ast::Literal(l) => l,
                        _ => return Err(self.err("a class escape cannot end a range")),
                    },
                    h => h,
                };
                if (lo as u32) > (hi as u32) {
                    return Err(self.err(format!("inverted range {lo:?}-{hi:?}")));
                }
                items.push(ClassItem::Range(lo, hi));
            } else {
                items.push(ClassItem::Single(lo));
            }
        }
        if items.is_empty() {
            return Err(self.err("empty character class"));
        }
        Ok(Ast::Class { negated, items })
    }
}

/// Sorted, disjoint, inclusive code-point ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CharSet(Vec<(u32, u32)>);

const MAX_CP: u32 = 0x10FFFF;

impl CharSet {
    fn any() -> Self {
        CharSet(vec![(0, MAX_CP)])
    }

    fn from_ranges(mut ranges: Vec<(u32, u32)>) -> Self {
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (lo, hi) in ranges {
            match merged.last_mut() {
                Some((_, phi)) if lo <= phi.saturating_add(1) => *phi = (*phi).max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        CharSet(merged)
    }

    fn complement(&self) -> Self {
        let mut out = Vec::new();
        let mut next = 0u32;
        for &(lo, hi) in &self.0 {
            if lo > next {
                out.push((next, lo - 1));
            }
            next = hi.saturating_add(1);
        }
        if next <= MAX_CP {
            out.push((next, MAX_CP));
        }
        CharSet(out)
    }

    fn contains(&self, c: char) -> bool {
        let cp = c as u32;
        self.0
            .binary_search_by(|&(lo, hi)| {
                if cp < lo {
                    std::cmp::Ordering::Greater
                } else if cp > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

fn perl_ranges(p: Perl) -> Vec<(u32, u32)> {
    // ASCII definitions (v1): \d=[0-9], \w=[0-9A-Za-z_], \s=[ \t\n\r\f\x0B].
    let positive = |p: Perl| -> Vec<(u32, u32)> {
        match p {
            Perl::Digit | Perl::NotDigit => vec![('0' as u32, '9' as u32)],
            Perl::Word | Perl::NotWord => vec![
                ('0' as u32, '9' as u32),
                ('A' as u32, 'Z' as u32),
                ('_' as u32, '_' as u32),
                ('a' as u32, 'z' as u32),
            ],
            Perl::Space | Perl::NotSpace => vec![(0x9, 0xD), (0x20, 0x20)],
        }
    };
    match p {
        Perl::Digit | Perl::Word | Perl::Space => positive(p),
        Perl::NotDigit | Perl::NotWord | Perl::NotSpace => {
            CharSet::from_ranges(positive(p)).complement().0
        }
    }
}

fn class_charset(negated: bool, items: &[ClassItem]) -> CharSet {
    let mut ranges = Vec::new();
    for item in items {
        match item {
            ClassItem::Single(c) => ranges.push((*c as u32, *c as u32)),
            ClassItem::Range(lo, hi) => ranges.push((*lo as u32, *hi as u32)),
            ClassItem::Perl(p) => ranges.extend(perl_ranges(*p)),
        }
    }
    let set = CharSet::from_ranges(ranges);
    if negated { set.complement() } else { set }
}

#[derive(Debug, Clone, Default)]
struct State {
    trans: Vec<(CharSet, usize)>,
    eps: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Nfa {
    states: Vec<State>,
    start: usize,
    accept: usize,
}

impl Nfa {
    fn compile(ast: &Ast) -> Result<Nfa, String> {
        let mut states = Vec::new();
        let mut b = Builder {
            states: &mut states,
        };
        let (start, accept) = b.build(ast)?;
        Ok(Nfa {
            states,
            start,
            accept,
        })
    }

    fn is_match(&self, input: &str) -> bool {
        let mut current = vec![false; self.states.len()];
        self.closure(self.start, &mut current);
        for c in input.chars() {
            let mut next = vec![false; self.states.len()];
            let mut any = false;
            for (i, active) in current.iter().enumerate() {
                if !active {
                    continue;
                }
                for (set, to) in &self.states[i].trans {
                    if set.contains(c) && !next[*to] {
                        self.closure(*to, &mut next);
                        any = true;
                    }
                }
            }
            if !any {
                return false;
            }
            current = next;
        }
        current[self.accept]
    }

    fn closure(&self, state: usize, set: &mut [bool]) {
        if set[state] {
            return;
        }
        set[state] = true;
        let mut stack = vec![state];
        while let Some(s) = stack.pop() {
            for &e in &self.states[s].eps {
                if !set[e] {
                    set[e] = true;
                    stack.push(e);
                }
            }
        }
    }
}

struct Builder<'a> {
    states: &'a mut Vec<State>,
}

impl Builder<'_> {
    fn state(&mut self) -> Result<usize, String> {
        if self.states.len() >= MAX_NFA_STATES {
            return Err(format!(
                "pattern compiles to more than {MAX_NFA_STATES} states; simplify nested repetitions"
            ));
        }
        self.states.push(State::default());
        Ok(self.states.len() - 1)
    }

    /// Thompson fragment: returns (start, accept).
    fn build(&mut self, ast: &Ast) -> Result<(usize, usize), String> {
        match ast {
            Ast::Empty => {
                let s = self.state()?;
                Ok((s, s))
            }
            Ast::Literal(c) => self.leaf(CharSet::from_ranges(vec![(*c as u32, *c as u32)])),
            Ast::Dot => self.leaf(CharSet::any()),
            Ast::Class { negated, items } => self.leaf(class_charset(*negated, items)),
            Ast::Concat(parts) => {
                let mut cur: Option<(usize, usize)> = None;
                for p in parts {
                    let (ps, pa) = self.build(p)?;
                    cur = Some(match cur {
                        None => (ps, pa),
                        Some((s, a)) => {
                            self.states[a].eps.push(ps);
                            (s, pa)
                        }
                    });
                }
                Ok(cur.expect("Concat is non-empty"))
            }
            Ast::Alt(branches) => {
                let start = self.state()?;
                let accept = self.state()?;
                for b in branches {
                    let (bs, ba) = self.build(b)?;
                    self.states[start].eps.push(bs);
                    self.states[ba].eps.push(accept);
                }
                Ok((start, accept))
            }
            Ast::Repeat { node, min, max } => match max {
                None => {
                    // min copies, then a loop.
                    let (ls, la) = self.build(node)?;
                    self.states[la].eps.push(ls);
                    let out = self.state()?;
                    self.states[la].eps.push(out);
                    if *min == 0 {
                        self.states[ls].eps.push(out);
                        Ok((ls, out))
                    } else {
                        let mut frag: Option<(usize, usize)> = None;
                        for _ in 1..*min {
                            let (s, a) = self.build(node)?;
                            frag = Some(match frag {
                                None => (s, a),
                                Some((fs, fa)) => {
                                    self.states[fa].eps.push(s);
                                    (fs, a)
                                }
                            });
                        }
                        match frag {
                            None => Ok((ls, out)),
                            Some((fs, fa)) => {
                                self.states[fa].eps.push(ls);
                                Ok((fs, out))
                            }
                        }
                    }
                }
                Some(max) => {
                    // min required copies, then (max-min) optional ones.
                    let start = self.state()?;
                    let out = self.state()?;
                    let mut tail = start;
                    for i in 0..*max {
                        if i >= *min {
                            self.states[tail].eps.push(out);
                        }
                        let (s, a) = self.build(node)?;
                        self.states[tail].eps.push(s);
                        tail = a;
                    }
                    self.states[tail].eps.push(out);
                    Ok((start, out))
                }
            },
        }
    }

    fn leaf(&mut self, set: CharSet) -> Result<(usize, usize), String> {
        let s = self.state()?;
        let a = self.state()?;
        self.states[s].trans.push((set, a));
        Ok((s, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, s: &str) -> bool {
        Regex::parse(p).unwrap().is_match(s)
    }

    #[test]
    fn literals_and_anchoring() {
        assert!(m("abc", "abc"));
        assert!(!m("abc", "xabc"));
        assert!(!m("abc", "abcx"));
        assert!(m("", ""));
        assert!(!m("", "a"));
    }

    #[test]
    fn classes_and_escapes() {
        assert!(m(r"[a-c]+", "abcba"));
        assert!(!m(r"[^a-c]", "b"));
        assert!(m(r"[^a-c]", "d"));
        assert!(m(r"\d{3}-\d{2}", "123-45"));
        assert!(m(r"\w+\s\w+", "hello world"));
        assert!(m(r"\.", "."));
        assert!(!m(r"\.", "a"));
        assert!(m(r"[\d-]+", "1-2"));
        assert!(m(r"\S+", "x_y"));
    }

    #[test]
    fn alternation_grouping_quantifiers() {
        assert!(m("(ab|cd)*", "abcdab"));
        assert!(m("a{2,4}", "aaa"));
        assert!(!m("a{2,4}", "a"));
        assert!(!m("a{2,4}", "aaaaa"));
        assert!(m("colou?r", "color"));
        assert!(m("colou?r", "colour"));
        assert!(m("a+", "aaaa"));
        assert!(!m("a+", ""));
    }

    #[test]
    fn rejections() {
        for bad in [
            r"a(?=b)",
            r"a*?",
            r"^a",
            r"a$",
            r"a{2,}",
            r"a{300}",
            r"(((((((((a)))))))))",
            r"a\1",
            "[z-a]",
            "[]",
        ] {
            assert!(Regex::parse(bad).is_err(), "expected rejection: {bad}");
        }
        assert!(Regex::parse(&"a".repeat(513)).is_err());
    }

    #[test]
    fn pathological_pattern_is_linear() {
        // (a|a)* against a^n + b: catastrophic for backtrackers, linear here.
        let r = Regex::parse("(a|a)*").unwrap();
        let mut s = "a".repeat(2000);
        assert!(r.is_match(&s));
        s.push('b');
        assert!(!r.is_match(&s));
    }

    #[test]
    fn structural_equality() {
        assert_eq!(Regex::parse("a|b").unwrap(), Regex::parse("a|b").unwrap());
        assert_ne!(Regex::parse("a|b").unwrap(), Regex::parse("b|a").unwrap());
    }

    #[test]
    fn nested_counted_repetition_hits_state_cap() {
        assert!(Regex::parse("((((a{256}){256}){256}){256})").is_err());
    }

    #[test]
    fn unicode_literals() {
        assert!(m("héllo", "héllo"));
        assert!(m(".", "é"));
        assert!(m("[α-ω]+", "αβγω"));
    }
}
