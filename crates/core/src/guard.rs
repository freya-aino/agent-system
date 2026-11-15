//! Guard expression DSL (`contracts/specs/schema.md`, `guard_dsl: 1`).
//!
//! Total by construction: no loops, no recursion, no calls, no allocation beyond the
//! expression tree. Grammar as pinned — note there is no grouping parenthesis in the
//! expression grammar; precedence and the ternary are the only structure.
//!
//! A guard that throws is impossible; a guard that cannot be evaluated (unwritten
//! channel, null propagation, division by a runtime zero) yields `false`.

use std::collections::BTreeMap;

use agent_contracts::Diagnostic;
use serde_json::Value;

use crate::regex::Regex;
use crate::schema::Schema;

pub const GUARD_DSL_VERSION: u32 = 1;

const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 512;

#[derive(Debug, Clone, PartialEq)]
pub struct Guard {
    source: String,
    ast: Expr,
}

impl Guard {
    pub fn source(&self) -> &str {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Lit(Lit),
    Path(Path),
    Unop(Unop, Box<Expr>),
    Binop(Binop, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
enum Lit {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Array(Vec<Lit>),
}

impl Lit {
    fn to_value(&self) -> Value {
        match self {
            Lit::Null => Value::Null,
            Lit::Bool(b) => Value::Bool(*b),
            Lit::Num(n) => serde_json::Number::from_f64(*n)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            Lit::Str(s) => Value::String(s.clone()),
            Lit::Array(items) => Value::Array(items.iter().map(Lit::to_value).collect()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Path {
    State { channel: String, segs: Vec<Seg> },
    Counter(Counter),
    Input { segs: Vec<Seg> },
}

#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Field(String),
    Index(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Counter {
    Iteration,
    Retries,
    Errors,
    Tokens,
    Cost,
    WallMs,
    Hops,
}

impl Counter {
    fn parse(name: &str) -> Option<Counter> {
        Some(match name {
            "iteration" => Counter::Iteration,
            "retries" => Counter::Retries,
            "errors" => Counter::Errors,
            "tokens" => Counter::Tokens,
            "cost" => Counter::Cost,
            "wall_ms" => Counter::WallMs,
            "hops" => Counter::Hops,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unop {
    Not,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binop {
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Func {
    Len,
    Has,
    IsNull,
    StartsWith,
    EndsWith,
    Contains,
    Matches,
    Min,
    Max,
    Abs,
    Floor,
    Ceil,
    Clamp,
    ToNumber,
    ToString,
    In,
}

impl Func {
    fn parse(name: &str) -> Option<Func> {
        Some(match name {
            "len" => Func::Len,
            "has" => Func::Has,
            "is_null" => Func::IsNull,
            "starts_with" => Func::StartsWith,
            "ends_with" => Func::EndsWith,
            "contains" => Func::Contains,
            "matches" => Func::Matches,
            "min" => Func::Min,
            "max" => Func::Max,
            "abs" => Func::Abs,
            "floor" => Func::Floor,
            "ceil" => Func::Ceil,
            "clamp" => Func::Clamp,
            "to_number" => Func::ToNumber,
            "to_string" => Func::ToString,
            "in" => Func::In,
            _ => return None,
        })
    }

    fn arity(self) -> usize {
        match self {
            Func::Len
            | Func::IsNull
            | Func::Abs
            | Func::Floor
            | Func::Ceil
            | Func::ToNumber
            | Func::ToString => 1,
            Func::Has
            | Func::StartsWith
            | Func::EndsWith
            | Func::Contains
            | Func::Matches
            | Func::Min
            | Func::Max
            | Func::In => 2,
            Func::Clamp => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    Sym(&'static str),
}

fn lex(src: &str) -> Result<Vec<Tok>, Diagnostic> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    let err = |i: usize, m: String| {
        Diagnostic::error("invalid_guard", "$", format!("at offset {i}: {m}"))
    };
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                toks.push(Tok::Ident(chars[start..i].iter().collect()));
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && matches!(chars[i - 1], 'e' | 'E')))
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let n: f64 = s
                    .parse()
                    .map_err(|_| err(start, format!("malformed number {s:?}")))?;
                if !n.is_finite() {
                    return Err(err(start, format!("number {s:?} overflows a double")));
                }
                toks.push(Tok::Num(n));
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                loop {
                    match chars.get(i) {
                        None => return Err(err(i, "unterminated string".into())),
                        Some('"') => {
                            i += 1;
                            break;
                        }
                        Some('\\') => {
                            i += 1;
                            let esc = chars
                                .get(i)
                                .ok_or_else(|| err(i, "dangling escape".into()))?;
                            s.push(match esc {
                                '"' => '"',
                                '\\' => '\\',
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                other => {
                                    return Err(err(i, format!("unsupported escape '\\{other}'")));
                                }
                            });
                            i += 1;
                        }
                        Some(c) => {
                            s.push(*c);
                            i += 1;
                        }
                    }
                }
                toks.push(Tok::Str(s));
            }
            _ => {
                let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
                let sym = match two.as_str() {
                    "&&" | "||" | "==" | "!=" | "<=" | ">=" => Some(match two.as_str() {
                        "&&" => "&&",
                        "||" => "||",
                        "==" => "==",
                        "!=" => "!=",
                        "<=" => "<=",
                        _ => ">=",
                    }),
                    _ => None,
                };
                if let Some(s) = sym {
                    toks.push(Tok::Sym(s));
                    i += 2;
                } else {
                    let one = match c {
                        '!' => "!",
                        '-' => "-",
                        '+' => "+",
                        '*' => "*",
                        '/' => "/",
                        '%' => "%",
                        '<' => "<",
                        '>' => ">",
                        '?' => "?",
                        ':' => ":",
                        '(' => "(",
                        ')' => ")",
                        '[' => "[",
                        ']' => "]",
                        ',' => ",",
                        '.' => ".",
                        other => return Err(err(i, format!("unexpected character {other:?}"))),
                    };
                    toks.push(Tok::Sym(one));
                    i += 1;
                }
            }
        }
    }
    Ok(toks)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub fn parse_guard(src: &str) -> Result<Guard, Vec<Diagnostic>> {
    let toks = lex(src).map_err(|d| vec![d])?;
    let mut p = GuardParser {
        toks,
        pos: 0,
        nodes: 0,
    };
    let ast = p.ternary(0).map_err(|d| vec![d])?;
    if p.pos < p.toks.len() {
        return Err(vec![Diagnostic::error(
            "invalid_guard",
            "$",
            format!("trailing tokens after expression (token {})", p.pos),
        )]);
    }
    Ok(Guard {
        source: src.to_owned(),
        ast,
    })
}

struct GuardParser {
    toks: Vec<Tok>,
    pos: usize,
    nodes: usize,
}

impl GuardParser {
    fn err(&self, m: impl Into<String>) -> Diagnostic {
        Diagnostic::error(
            "invalid_guard",
            "$",
            format!("at token {}: {}", self.pos, m.into()),
        )
    }

    fn node(&mut self, depth: usize, e: Expr) -> Result<Expr, Diagnostic> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(self.err(format!("expression exceeds {MAX_NODES} nodes")));
        }
        if depth > MAX_DEPTH {
            return Err(self.err(format!("expression deeper than {MAX_DEPTH}")));
        }
        Ok(e)
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_sym(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_sym(&mut self, s: &str) -> Result<(), Diagnostic> {
        if self.eat_sym(s) {
            Ok(())
        } else {
            Err(self.err(format!("expected {s:?}")))
        }
    }

    fn ternary(&mut self, depth: usize) -> Result<Expr, Diagnostic> {
        let cond = self.binary(0, depth + 1)?;
        if self.eat_sym("?") {
            let t = self.ternary(depth + 1)?;
            self.expect_sym(":")?;
            let e = self.ternary(depth + 1)?;
            return self.node(
                depth,
                Expr::Ternary(Box::new(cond), Box::new(t), Box::new(e)),
            );
        }
        Ok(cond)
    }

    /// Precedence climbing over the binop tiers.
    fn binary(&mut self, tier: usize, depth: usize) -> Result<Expr, Diagnostic> {
        const TIERS: &[&[(&str, Binop)]] = &[
            &[("||", Binop::Or)],
            &[("&&", Binop::And)],
            &[("==", Binop::Eq), ("!=", Binop::Ne)],
            &[
                ("<=", Binop::Le),
                (">=", Binop::Ge),
                ("<", Binop::Lt),
                (">", Binop::Gt),
            ],
            &[("+", Binop::Add), ("-", Binop::Sub)],
            &[("*", Binop::Mul), ("/", Binop::Div), ("%", Binop::Rem)],
        ];
        if tier >= TIERS.len() {
            return self.unary(depth);
        }
        let mut lhs = self.binary(tier + 1, depth + 1)?;
        'outer: loop {
            for (sym, op) in TIERS[tier] {
                if self.eat_sym(sym) {
                    let rhs = self.binary(tier + 1, depth + 1)?;
                    if matches!(op, Binop::Div | Binop::Rem)
                        && let Expr::Lit(Lit::Num(n)) = rhs
                        && n == 0.0
                    {
                        return Err(self.err("division or % by a literal zero"));
                    }
                    lhs = self.node(depth, Expr::Binop(*op, Box::new(lhs), Box::new(rhs)))?;
                    continue 'outer;
                }
            }
            return Ok(lhs);
        }
    }

    fn unary(&mut self, depth: usize) -> Result<Expr, Diagnostic> {
        if self.eat_sym("!") {
            let e = self.unary(depth + 1)?;
            return self.node(depth, Expr::Unop(Unop::Not, Box::new(e)));
        }
        if self.eat_sym("-") {
            let e = self.unary(depth + 1)?;
            // Fold a negated numeric literal so `-1` is a literal, not an op.
            if let Expr::Lit(Lit::Num(n)) = e {
                return self.node(depth, Expr::Lit(Lit::Num(-n)));
            }
            return self.node(depth, Expr::Unop(Unop::Neg, Box::new(e)));
        }
        self.primary(depth)
    }

    fn primary(&mut self, depth: usize) -> Result<Expr, Diagnostic> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                self.node(depth, Expr::Lit(Lit::Num(n)))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                self.node(depth, Expr::Lit(Lit::Str(s)))
            }
            Some(Tok::Sym("[")) => {
                self.pos += 1;
                let mut items = Vec::new();
                if !self.eat_sym("]") {
                    loop {
                        items.push(self.literal()?);
                        if self.eat_sym("]") {
                            break;
                        }
                        self.expect_sym(",")?;
                    }
                }
                self.node(depth, Expr::Lit(Lit::Array(items)))
            }
            Some(Tok::Ident(id)) => {
                self.pos += 1;
                match id.as_str() {
                    "true" => self.node(depth, Expr::Lit(Lit::Bool(true))),
                    "false" => self.node(depth, Expr::Lit(Lit::Bool(false))),
                    "null" => self.node(depth, Expr::Lit(Lit::Null)),
                    "state" => {
                        self.expect_sym(".")?;
                        let channel = self.ident()?;
                        let segs = self.segs()?;
                        self.node(depth, Expr::Path(Path::State { channel, segs }))
                    }
                    "self" => {
                        self.expect_sym(".")?;
                        let name = self.ident()?;
                        let c = Counter::parse(&name)
                            .ok_or_else(|| self.err(format!("unknown self counter {name:?}")))?;
                        self.node(depth, Expr::Path(Path::Counter(c)))
                    }
                    "input" => {
                        let segs = self.segs()?;
                        self.node(depth, Expr::Path(Path::Input { segs }))
                    }
                    name => {
                        let f = Func::parse(name)
                            .ok_or_else(|| self.err(format!("unknown identifier {name:?}")))?;
                        self.expect_sym("(")?;
                        let mut args = Vec::new();
                        if !self.eat_sym(")") {
                            loop {
                                args.push(self.ternary(depth + 1)?);
                                if self.eat_sym(")") {
                                    break;
                                }
                                self.expect_sym(",")?;
                            }
                        }
                        if args.len() != f.arity() {
                            return Err(self.err(format!(
                                "{name} takes {} argument(s), got {}",
                                f.arity(),
                                args.len()
                            )));
                        }
                        if f == Func::Matches {
                            let Expr::Lit(Lit::Str(pattern)) = &args[1] else {
                                return Err(self.err("matches requires a string literal pattern"));
                            };
                            Regex::parse_at(pattern, "$").map_err(|d| {
                                Diagnostic::error(
                                    "invalid_guard",
                                    "$",
                                    format!("matches pattern: {}", d.message),
                                )
                            })?;
                        }
                        if f == Func::In && !matches!(&args[1], Expr::Lit(Lit::Array(_))) {
                            return Err(self.err("in requires an array literal"));
                        }
                        self.node(depth, Expr::Call(f, args))
                    }
                }
            }
            _ => Err(self.err("expected an expression")),
        }
    }

    fn literal(&mut self) -> Result<Lit, Diagnostic> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(Lit::Num(n))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Lit::Str(s))
            }
            Some(Tok::Ident(id)) if id == "true" || id == "false" || id == "null" => {
                self.pos += 1;
                Ok(match id.as_str() {
                    "true" => Lit::Bool(true),
                    "false" => Lit::Bool(false),
                    _ => Lit::Null,
                })
            }
            Some(Tok::Sym("-")) => {
                self.pos += 1;
                match self.peek().cloned() {
                    Some(Tok::Num(n)) => {
                        self.pos += 1;
                        Ok(Lit::Num(-n))
                    }
                    _ => Err(self.err("expected a number after '-'")),
                }
            }
            Some(Tok::Sym("[")) => {
                self.pos += 1;
                let mut items = Vec::new();
                if !self.eat_sym("]") {
                    loop {
                        items.push(self.literal()?);
                        if self.eat_sym("]") {
                            break;
                        }
                        self.expect_sym(",")?;
                    }
                }
                Ok(Lit::Array(items))
            }
            _ => Err(self.err("expected a literal")),
        }
    }

    fn ident(&mut self) -> Result<String, Diagnostic> {
        match self.peek().cloned() {
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                Ok(s)
            }
            _ => Err(self.err("expected an identifier")),
        }
    }

    fn segs(&mut self) -> Result<Vec<Seg>, Diagnostic> {
        let mut segs = Vec::new();
        loop {
            if self.eat_sym(".") {
                segs.push(Seg::Field(self.ident()?));
            } else if self.eat_sym("[") {
                let idx = match self.peek().cloned() {
                    Some(Tok::Num(n)) if n.fract() == 0.0 && n >= 0.0 => {
                        self.pos += 1;
                        n as u64
                    }
                    _ => return Err(self.err("expected a non-negative integer index")),
                };
                self.expect_sym("]")?;
                segs.push(Seg::Index(idx));
            } else {
                return Ok(segs);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Typechecking
// ---------------------------------------------------------------------------

/// What typechecking resolves paths against: declared channel schemas and the source
/// node's output schema.
pub struct GuardContext<'a> {
    pub channels: &'a BTreeMap<String, Schema>,
    pub input: &'a Schema,
}

#[derive(Debug, Clone, PartialEq)]
enum Ty {
    Null,
    Bool,
    Num,
    Str,
    Arr(Box<Ty>),
    Obj(BTreeMap<String, Schema>),
    /// A type the checker cannot refine (mixed enum, oneOf): only `==`, `!=`,
    /// `is_null` and `to_string` apply.
    Opaque,
}

fn schema_ty(s: &Schema) -> Ty {
    match s {
        Schema::Null => Ty::Null,
        Schema::Boolean => Ty::Bool,
        Schema::Integer { .. } | Schema::Number { .. } => Ty::Num,
        Schema::Str { .. } => Ty::Str,
        Schema::Array { items, .. } => Ty::Arr(Box::new(schema_ty(items))),
        Schema::Object { properties, .. } => Ty::Obj(properties.clone()),
        Schema::Const(v) => value_ty(v),
        Schema::Enum(vs) => {
            let tys: Vec<Ty> = vs.iter().map(value_ty).collect();
            if tys.windows(2).all(|w| w[0] == w[1]) {
                tys.into_iter().next().unwrap_or(Ty::Opaque)
            } else {
                Ty::Opaque
            }
        }
        Schema::OneOf(branches) => {
            let tys: Vec<Ty> = branches.iter().map(schema_ty).collect();
            if tys.windows(2).all(|w| w[0] == w[1]) {
                tys.into_iter().next().unwrap_or(Ty::Opaque)
            } else {
                Ty::Opaque
            }
        }
    }
}

fn value_ty(v: &Value) -> Ty {
    match v {
        Value::Null => Ty::Null,
        Value::Bool(_) => Ty::Bool,
        Value::Number(_) => Ty::Num,
        Value::String(_) => Ty::Str,
        Value::Array(_) | Value::Object(_) => Ty::Opaque,
    }
}

pub fn typecheck(guard: &Guard, ctx: &GuardContext) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    check(&guard.ast, ctx, &mut diags);
    if let Some(ty) = infer(&guard.ast, ctx)
        && ty != Ty::Bool
    {
        diags.push(Diagnostic::error(
            "invalid_guard",
            "$",
            "a guard must evaluate to a boolean",
        ));
    }
    if diags.is_empty() { Ok(()) } else { Err(diags) }
}

fn check(e: &Expr, ctx: &GuardContext, diags: &mut Vec<Diagnostic>) {
    // infer() reports into diags via a second pass; keep one entry point.
    let before = diags.len();
    if infer_with(e, ctx, diags).is_none() && diags.len() == before {
        diags.push(Diagnostic::error(
            "invalid_guard",
            "$",
            "expression does not typecheck",
        ));
    }
}

fn infer(e: &Expr, ctx: &GuardContext) -> Option<Ty> {
    infer_with(e, ctx, &mut Vec::new())
}

fn infer_with(e: &Expr, ctx: &GuardContext, diags: &mut Vec<Diagnostic>) -> Option<Ty> {
    let fail = |diags: &mut Vec<Diagnostic>, m: String| {
        diags.push(Diagnostic::error("invalid_guard", "$", m));
        None
    };
    match e {
        Expr::Lit(l) => Some(lit_ty(l)),
        Expr::Path(p) => path_ty(p, ctx, diags),
        Expr::Unop(op, inner) => {
            let t = infer_with(inner, ctx, diags)?;
            match (op, &t) {
                (Unop::Not, Ty::Bool) => Some(Ty::Bool),
                (Unop::Neg, Ty::Num) => Some(Ty::Num),
                _ => fail(diags, format!("{op:?} applied to {t:?}")),
            }
        }
        Expr::Binop(op, l, r) => {
            let lt = infer_with(l, ctx, diags)?;
            let rt = infer_with(r, ctx, diags)?;
            match op {
                Binop::And | Binop::Or => {
                    if lt == Ty::Bool && rt == Ty::Bool {
                        Some(Ty::Bool)
                    } else {
                        fail(
                            diags,
                            format!("{op:?} needs booleans, got {lt:?} and {rt:?}"),
                        )
                    }
                }
                Binop::Eq | Binop::Ne => {
                    if lt == rt
                        || lt == Ty::Null
                        || rt == Ty::Null
                        || lt == Ty::Opaque
                        || rt == Ty::Opaque
                    {
                        Some(Ty::Bool)
                    } else {
                        fail(diags, format!("comparing {lt:?} with {rt:?}"))
                    }
                }
                Binop::Lt | Binop::Le | Binop::Gt | Binop::Ge => {
                    if lt == Ty::Num && rt == Ty::Num {
                        Some(Ty::Bool)
                    } else {
                        fail(
                            diags,
                            format!("{op:?} needs numbers, got {lt:?} and {rt:?}"),
                        )
                    }
                }
                Binop::Add | Binop::Sub | Binop::Mul | Binop::Div | Binop::Rem => {
                    if lt == Ty::Num && rt == Ty::Num {
                        Some(Ty::Num)
                    } else {
                        fail(
                            diags,
                            format!("{op:?} needs numbers, got {lt:?} and {rt:?}"),
                        )
                    }
                }
            }
        }
        Expr::Ternary(c, t, f) => {
            let ct = infer_with(c, ctx, diags)?;
            if ct != Ty::Bool {
                return fail(
                    diags,
                    format!("ternary condition must be a boolean, got {ct:?}"),
                );
            }
            let tt = infer_with(t, ctx, diags)?;
            let ft = infer_with(f, ctx, diags)?;
            if tt == ft {
                Some(tt)
            } else if tt == Ty::Null {
                Some(ft)
            } else if ft == Ty::Null {
                Some(tt)
            } else {
                fail(
                    diags,
                    format!("ternary branches disagree: {tt:?} vs {ft:?}"),
                )
            }
        }
        Expr::Call(f, args) => {
            let tys: Option<Vec<Ty>> = args.iter().map(|a| infer_with(a, ctx, diags)).collect();
            let tys = tys?;
            match f {
                Func::Len => match &tys[0] {
                    Ty::Str | Ty::Arr(_) => Some(Ty::Num),
                    t => fail(diags, format!("len of {t:?}")),
                },
                Func::Has => match (&tys[0], &tys[1]) {
                    (Ty::Obj(_) | Ty::Opaque, Ty::Str) => Some(Ty::Bool),
                    (a, b) => fail(diags, format!("has({a:?}, {b:?})")),
                },
                Func::IsNull => Some(Ty::Bool),
                Func::StartsWith | Func::EndsWith => match (&tys[0], &tys[1]) {
                    (Ty::Str, Ty::Str) => Some(Ty::Bool),
                    (a, b) => fail(diags, format!("{f:?}({a:?}, {b:?})")),
                },
                Func::Contains => match (&tys[0], &tys[1]) {
                    (Ty::Str, Ty::Str) => Some(Ty::Bool),
                    (Ty::Arr(_), _) => Some(Ty::Bool),
                    (a, b) => fail(diags, format!("contains({a:?}, {b:?})")),
                },
                Func::Matches => match (&tys[0], &tys[1]) {
                    (Ty::Str, Ty::Str) => Some(Ty::Bool),
                    (a, _) => fail(diags, format!("matches on {a:?}")),
                },
                Func::Min | Func::Max => match (&tys[0], &tys[1]) {
                    (Ty::Num, Ty::Num) => Some(Ty::Num),
                    (a, b) => fail(diags, format!("{f:?}({a:?}, {b:?})")),
                },
                Func::Abs | Func::Floor | Func::Ceil => match &tys[0] {
                    Ty::Num => Some(Ty::Num),
                    t => fail(diags, format!("{f:?} of {t:?}")),
                },
                Func::Clamp => {
                    if tys.iter().all(|t| *t == Ty::Num) {
                        Some(Ty::Num)
                    } else {
                        fail(diags, format!("clamp needs numbers, got {tys:?}"))
                    }
                }
                Func::ToNumber => match &tys[0] {
                    Ty::Str | Ty::Num | Ty::Bool => Some(Ty::Num),
                    t => fail(diags, format!("to_number of {t:?}")),
                },
                Func::ToString => match &tys[0] {
                    Ty::Str | Ty::Num | Ty::Bool | Ty::Null | Ty::Opaque => Some(Ty::Str),
                    t => fail(diags, format!("to_string of {t:?}")),
                },
                Func::In => match (&tys[0], &tys[1]) {
                    (Ty::Str | Ty::Num | Ty::Bool | Ty::Null, Ty::Arr(_)) => Some(Ty::Bool),
                    (a, b) => fail(diags, format!("in({a:?}, {b:?})")),
                },
            }
        }
    }
}

fn lit_ty(l: &Lit) -> Ty {
    match l {
        Lit::Null => Ty::Null,
        Lit::Bool(_) => Ty::Bool,
        Lit::Num(_) => Ty::Num,
        Lit::Str(_) => Ty::Str,
        Lit::Array(items) => {
            let tys: Vec<Ty> = items.iter().map(lit_ty).collect();
            if !tys.is_empty() && tys.windows(2).all(|w| w[0] == w[1]) {
                Ty::Arr(Box::new(tys.into_iter().next().expect("non-empty")))
            } else {
                Ty::Arr(Box::new(Ty::Opaque))
            }
        }
    }
}

fn path_ty(p: &Path, ctx: &GuardContext, diags: &mut Vec<Diagnostic>) -> Option<Ty> {
    match p {
        Path::Counter(_) => Some(Ty::Num),
        Path::State { channel, segs } => {
            let Some(schema) = ctx.channels.get(channel) else {
                diags.push(Diagnostic::error(
                    "invalid_guard",
                    "$",
                    format!("unknown channel {channel:?}"),
                ));
                return None;
            };
            nav_ty(schema, segs, &format!("state.{channel}"), diags)
        }
        Path::Input { segs } => nav_ty(ctx.input, segs, "input", diags),
    }
}

fn nav_ty(schema: &Schema, segs: &[Seg], what: &str, diags: &mut Vec<Diagnostic>) -> Option<Ty> {
    let mut cur = schema;
    for seg in segs {
        match (seg, cur) {
            (Seg::Field(f), Schema::Object { properties, .. }) => match properties.get(f) {
                Some(s) => cur = s,
                None => {
                    diags.push(Diagnostic::error(
                        "invalid_guard",
                        "$",
                        format!("{what} has no field {f:?}"),
                    ));
                    return None;
                }
            },
            (Seg::Index(_), Schema::Array { items, .. }) => cur = items,
            (seg, _) => {
                diags.push(Diagnostic::error(
                    "invalid_guard",
                    "$",
                    format!("cannot apply {seg:?} to {what}"),
                ));
                return None;
            }
        }
    }
    Some(schema_ty(cur))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct SelfCounters {
    pub iteration: u64,
    pub retries: u64,
    pub errors: u64,
    pub tokens: u64,
    pub cost: f64,
    pub wall_ms: u64,
    pub hops: u64,
}

pub struct GuardEnv<'a> {
    /// Written channels only; an unwritten channel is simply absent and makes the
    /// guard `false`.
    pub channels: &'a BTreeMap<String, Value>,
    pub counters: SelfCounters,
    pub input: &'a Value,
}

/// Terminates on every well-typed input and never panics; a `null` result is `false`.
pub fn eval_guard(guard: &Guard, env: &GuardEnv) -> bool {
    matches!(eval(&guard.ast, env), Some(Value::Bool(true)))
}

/// `None` is the DSL's `null`.
fn eval(e: &Expr, env: &GuardEnv) -> Option<Value> {
    match e {
        Expr::Lit(l) => match l.to_value() {
            Value::Null => None,
            v => Some(v),
        },
        Expr::Path(p) => eval_path(p, env),
        Expr::Unop(op, inner) => {
            let v = eval(inner, env)?;
            match (op, v) {
                (Unop::Not, Value::Bool(b)) => Some(Value::Bool(!b)),
                (Unop::Neg, Value::Number(n)) => num(-n.as_f64()?),
                _ => None,
            }
        }
        Expr::Binop(op, l, r) => {
            match op {
                // && and || are strict in both operands (the DSL is total, so there
                // is no cost to it) but null-tolerant: null acts as false.
                Binop::And => {
                    let a = truthy(eval(l, env));
                    let b = truthy(eval(r, env));
                    return Some(Value::Bool(a && b));
                }
                Binop::Or => {
                    let a = truthy(eval(l, env));
                    let b = truthy(eval(r, env));
                    return Some(Value::Bool(a || b));
                }
                Binop::Eq | Binop::Ne => {
                    let a = eval(l, env);
                    let b = eval(r, env);
                    let eq = a == b;
                    return Some(Value::Bool(if *op == Binop::Eq { eq } else { !eq }));
                }
                _ => {}
            }
            let a = eval(l, env)?.as_f64()?;
            let b = eval(r, env)?.as_f64()?;
            match op {
                Binop::Lt => Some(Value::Bool(a < b)),
                Binop::Le => Some(Value::Bool(a <= b)),
                Binop::Gt => Some(Value::Bool(a > b)),
                Binop::Ge => Some(Value::Bool(a >= b)),
                Binop::Add => num(a + b),
                Binop::Sub => num(a - b),
                Binop::Mul => num(a * b),
                // By a runtime zero they evaluate to null (and a null guard is false).
                Binop::Div => {
                    if b == 0.0 {
                        None
                    } else {
                        num(a / b)
                    }
                }
                Binop::Rem => {
                    if b == 0.0 {
                        None
                    } else {
                        num(a % b)
                    }
                }
                Binop::And | Binop::Or | Binop::Eq | Binop::Ne => unreachable!("handled above"),
            }
        }
        Expr::Ternary(c, t, f) => {
            if truthy(eval(c, env)) {
                eval(t, env)
            } else {
                eval(f, env)
            }
        }
        Expr::Call(f, args) => eval_call(*f, args, env),
    }
}

fn truthy(v: Option<Value>) -> bool {
    matches!(v, Some(Value::Bool(true)))
}

fn num(f: f64) -> Option<Value> {
    serde_json::Number::from_f64(f).map(Value::Number)
}

fn eval_path(p: &Path, env: &GuardEnv) -> Option<Value> {
    match p {
        Path::Counter(c) => num(match c {
            Counter::Iteration => env.counters.iteration as f64,
            Counter::Retries => env.counters.retries as f64,
            Counter::Errors => env.counters.errors as f64,
            Counter::Tokens => env.counters.tokens as f64,
            Counter::Cost => env.counters.cost,
            Counter::WallMs => env.counters.wall_ms as f64,
            Counter::Hops => env.counters.hops as f64,
        }),
        Path::State { channel, segs } => nav_value(env.channels.get(channel)?, segs),
        Path::Input { segs } => nav_value(env.input, segs),
    }
}

fn nav_value(root: &Value, segs: &[Seg]) -> Option<Value> {
    let mut cur = root;
    for seg in segs {
        cur = match seg {
            Seg::Field(f) => cur.as_object()?.get(f)?,
            Seg::Index(i) => cur.as_array()?.get(*i as usize)?,
        };
    }
    if cur.is_null() {
        None
    } else {
        Some(cur.clone())
    }
}

fn eval_call(f: Func, args: &[Expr], env: &GuardEnv) -> Option<Value> {
    if f == Func::IsNull {
        return Some(Value::Bool(eval(&args[0], env).is_none()));
    }
    if f == Func::Has {
        let obj = eval(&args[0], env)?;
        let key = eval(&args[1], env)?;
        return Some(Value::Bool(obj.as_object()?.contains_key(key.as_str()?)));
    }
    let vals: Option<Vec<Value>> = args.iter().map(|a| eval(a, env)).collect();
    let vals = vals?;
    match f {
        Func::Len => match &vals[0] {
            Value::String(s) => num(s.chars().count() as f64),
            Value::Array(a) => num(a.len() as f64),
            _ => None,
        },
        Func::StartsWith => Some(Value::Bool(
            vals[0].as_str()?.starts_with(vals[1].as_str()?),
        )),
        Func::EndsWith => Some(Value::Bool(vals[0].as_str()?.ends_with(vals[1].as_str()?))),
        Func::Contains => match &vals[0] {
            Value::String(s) => Some(Value::Bool(s.contains(vals[1].as_str()?))),
            Value::Array(a) => Some(Value::Bool(a.contains(&vals[1]))),
            _ => None,
        },
        Func::Matches => {
            // The pattern is a literal validated at parse time; recompilation here
            // keeps Guard free of interior mutability. Compiled size is bounded.
            let pattern = Regex::parse(vals[1].as_str()?).ok()?;
            Some(Value::Bool(pattern.is_match(vals[0].as_str()?)))
        }
        Func::Min => num(vals[0].as_f64()?.min(vals[1].as_f64()?)),
        Func::Max => num(vals[0].as_f64()?.max(vals[1].as_f64()?)),
        Func::Abs => num(vals[0].as_f64()?.abs()),
        Func::Floor => num(vals[0].as_f64()?.floor()),
        Func::Ceil => num(vals[0].as_f64()?.ceil()),
        Func::Clamp => {
            let (x, lo, hi) = (vals[0].as_f64()?, vals[1].as_f64()?, vals[2].as_f64()?);
            if lo > hi {
                return None;
            }
            num(x.clamp(lo, hi))
        }
        Func::ToNumber => match &vals[0] {
            Value::Number(n) => Some(Value::Number(n.clone())),
            Value::String(s) => s
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|f| f.is_finite())
                .and_then(num),
            Value::Bool(b) => num(if *b { 1.0 } else { 0.0 }),
            _ => None,
        },
        Func::ToString => match &vals[0] {
            Value::String(s) => Some(Value::String(s.clone())),
            Value::Number(n) => Some(Value::String(crate::canon::js_number(n.as_f64()?))),
            Value::Bool(b) => Some(Value::String(b.to_string())),
            v => Some(Value::String(v.to_string())),
        },
        Func::In => Some(Value::Bool(vals[1].as_array()?.contains(&vals[0]))),
        Func::IsNull | Func::Has => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx_schemas() -> BTreeMap<String, Schema> {
        let mut m = BTreeMap::new();
        m.insert(
            "review".to_owned(),
            crate::schema::parse_schema(&json!({
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["score"]
            }))
            .unwrap(),
        );
        m
    }

    fn input_schema() -> Schema {
        crate::schema::parse_schema(
            &json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": []}),
        )
        .unwrap()
    }

    fn well_typed(src: &str) -> Guard {
        let g = parse_guard(src).unwrap();
        let channels = ctx_schemas();
        let input = input_schema();
        typecheck(
            &g,
            &GuardContext {
                channels: &channels,
                input: &input,
            },
        )
        .unwrap();
        g
    }

    fn run(src: &str, channels: Value, input: Value, counters: SelfCounters) -> bool {
        let g = well_typed(src);
        let channels: BTreeMap<String, Value> = channels
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        eval_guard(
            &g,
            &GuardEnv {
                channels: &channels,
                counters,
                input: &input,
            },
        )
    }

    #[test]
    fn basic_evaluation() {
        let c = json!({"review": {"score": 0.8, "tags": ["ok"]}});
        assert!(run(
            "state.review.score > 0.5",
            c.clone(),
            json!({}),
            SelfCounters::default()
        ));
        assert!(!run(
            "state.review.score > 0.9",
            c.clone(),
            json!({}),
            SelfCounters::default()
        ));
        assert!(run(
            "contains(state.review.tags, \"ok\")",
            c.clone(),
            json!({}),
            SelfCounters::default()
        ));
        assert!(run(
            "len(state.review.tags) == 1",
            c,
            json!({}),
            SelfCounters::default()
        ));
    }

    #[test]
    fn unwritten_channel_is_false() {
        assert!(!run(
            "state.review.score > 0.0",
            json!({}),
            json!({}),
            SelfCounters::default()
        ));
    }

    #[test]
    fn self_counters_and_ternary() {
        let counters = SelfCounters {
            iteration: 3,
            tokens: 900,
            ..Default::default()
        };
        assert!(run(
            "self.iteration < 5 ? self.tokens < 1000 : false",
            json!({}),
            json!({}),
            counters
        ));
    }

    #[test]
    fn division_by_runtime_zero_is_false() {
        let c = json!({"review": {"score": 0.0}});
        // score is 0 at run time; the literal-zero case is rejected at parse time.
        assert!(!run(
            "1 / state.review.score > 0.0",
            c,
            json!({}),
            SelfCounters::default()
        ));
        assert!(parse_guard("1 / 0 > 2").is_err());
        assert!(parse_guard("1 % 0 > 2").is_err());
    }

    #[test]
    fn matches_and_input() {
        assert!(run(
            "input.ok == true",
            json!({}),
            json!({"ok": true}),
            SelfCounters::default()
        ));
        // Guard-source string escaping: the pattern `[a-z]+-\d+` is written with a
        // doubled backslash inside the guard string literal.
        let g = parse_guard("matches(\"abc-123\", \"[a-z]+-\\\\d+\")").unwrap();
        let channels = BTreeMap::new();
        assert!(eval_guard(
            &g,
            &GuardEnv {
                channels: &channels,
                counters: SelfCounters::default(),
                input: &Value::Null
            }
        ));
        assert!(parse_guard("matches(\"x\", \"(?=bad)\")").is_err());
    }

    #[test]
    fn typecheck_rejects() {
        let channels = ctx_schemas();
        let input = input_schema();
        let ctx = GuardContext {
            channels: &channels,
            input: &input,
        };
        for bad in [
            "state.unknown.x == 1",
            "state.review.nope == 1",
            "state.review.score && true",
            "len(state.review.score) > 0",
            "state.review.score + \"x\" == 1",
            "state.review.score", // not a boolean
        ] {
            let g = parse_guard(bad).unwrap();
            assert!(
                typecheck(&g, &ctx).is_err(),
                "expected typecheck rejection: {bad}"
            );
        }
    }

    #[test]
    fn limits_are_enforced() {
        let wide = format!("1 {} > 0", "+ 1".repeat(600));
        assert!(parse_guard(&wide).is_err());
        let deep = format!("{}true", "!".repeat(80));
        assert!(parse_guard(&deep).is_err());
    }

    #[test]
    fn in_and_null_handling() {
        assert!(run(
            "in(to_string(self.iteration), [\"0\", \"1\"])",
            json!({}),
            json!({}),
            SelfCounters::default()
        ));
        assert!(run(
            "is_null(to_number(\"not a number\"))",
            json!({}),
            json!({}),
            SelfCounters::default()
        ));
        assert!(!run(
            "to_number(\"nan\") > 0.0",
            json!({}),
            json!({}),
            SelfCounters::default()
        ));
    }
}
