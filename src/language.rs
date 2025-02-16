use std::fmt::{Display, self};

const PUNCTS: &[char] = &['(', ')', '{', '}', ';', ':', ',', '-', '>'];

type Int = i32;
type Float = f32;

trait Optionalise where Self: Sized {
    fn some(self) -> Option<Self> {
        Some(self)
    }
}

impl<T> Optionalise for T {}

trait Wrap where Self: Sized {
    fn wrap(self) -> Box<Self> {
        Box::new(self)
    }
}

impl<T> Wrap for T {}

#[derive(Debug)]
pub struct ParseError {
    err: ParseErrorTy,
    loc: Location,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} - ERROR : {}", self.loc, self.err)
    }
}

impl ParseError {
    fn from(token: &Token, ty: ParseErrorTy) -> Self {
        Self {
            loc: token.loc.clone(),
            err: ty,
        }
    }

    fn ty(ty: ParseErrorTy) -> Self {
        Self {
            loc: Location::default(),
            err: ty,
        }
    }
}

#[derive(Debug)]
pub enum ParseErrorTy {
    ExpectedTokenType(TokenType),
    ExpectedPunct(char),
    EmptyTokens,
    CannotParse(Vec<Token>),
    CannotParseTypeFrom(TokenType),
    ExpectedLiteralOfType(Type, TokenType),
}

impl Display for ParseErrorTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::ExpectedTokenType(tok) => write!(f, "Expected token of type: `{tok:?}`"),
            Self::ExpectedLiteralOfType(ty, tok) => write!(f, "Expected literal of type: `{ty:?}`, got `{tok:?}`"),
            Self::CannotParseTypeFrom(tokty) => write!(f, "Cannot parse type from token: `{tokty:?}`, expected identifier"),
            Self::ExpectedPunct(c) => write!(f, "Expected punctuation: `{c}`"),
            Self::EmptyTokens => write!(f, "Reached end of tokens or was passed an empty list"),
            Self::CannotParse(tokens) => write!(f, "Could not parse tokens: `{tokens:?}`"),
        }
    }
}

type ParseResult<T> = Result<T, ParseError>;

struct Ast {
    puncts:         &'static [char],
    keywords:       &'static [&'static str],
    string_enter:   char,
    string_exit:    char,
    keep_newlines:  bool,
}

impl Ast {
    fn parse(&self, text: impl ToString) -> Option<Vec<Token>> {
        let string = text.to_string();
        let chars = string.chars().collect::<Vec<char>>();

        let mut line = 1;
        let mut column_og = 1;
        let mut column = 1;

        let mut tokens: Vec<Token> = vec![];

        let mut buf = String::with_capacity(10);
        let mut in_string = false;

        let mut evaluate_char = |c| -> Option<()> {
            if in_string {
                if c == self.string_exit {
                    in_string = false;
                    tokens.push(Token::ty(TokenType::StringLiteral(buf.drain(..).collect::<_>()), line, column_og, column));
                    column_og = column;
                }
                else {
                    buf.push(c);
                }
            }
            else if c.is_whitespace() {
                if c == '\n' {
                    // TODO: extract this horrible logic :( i don't like repeating it
                    if !buf.trim().is_empty() {
                        tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
                        buf.clear();
                        column_og = column;
                    }
                    if self.keep_newlines { tokens.push(Token::new("\n", line, column_og, column)?); }
                    line += 1;
                    column = 0;
                }
                else if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
                    column_og = column;
                }
                buf.clear();
            }
            else if c == self.string_enter {
                in_string = true;
            }
            else if self.puncts.contains(&c) {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
                    buf.clear();
                    column_og = column;
                }
                tokens.push(Token::ty(TokenType::Punct(c), line, column_og, column));
                column_og = column;
            }
            else if c == '\0' {
                if !buf.trim().is_empty() {
                    tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
                    buf.clear();
                    column_og = column;
                }
                tokens.push(Token::ty(TokenType::Eof, line, column_og, column));
            }
            else {
                buf.push(c);
            }
            column += 1;
            Some(())
        };

        for c in chars {
            evaluate_char(c)?;
        }
        evaluate_char('\0')?;

        tokens = tokens.into_iter().map(|t| {
            if let TokenType::Ident(ref ident) = t.ty { 
                if self.keywords.contains(&ident.as_str()) { Token { ty: TokenType::Keyword(ident.to_string()), loc: t.loc } } 
                else { t }
            }
            else { t }
        }).collect();

        Some(tokens)
    }
}

#[derive(Debug, Clone)]
struct Token {
    ty: TokenType,
    loc: Location,
}

impl Token {
    fn new(text: &str, line: usize, column_start: usize, column_end: usize) -> Option<Self> {
        Some(Self {
            ty: TokenType::from_str(text)?,
            loc: Location::new(line, column_start, column_end),
        })
    }

    fn ty(ty: TokenType, line: usize, column_start: usize, column_end: usize) -> Self {
        Self {
            ty,
            loc: Location::new(line, column_start, column_end),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Location {
    line: usize,
    column_start: usize,
    column_end: usize,
}

impl Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}:{}", self.line, self.column_start)
    }
}

impl Default for Location {
    fn default() -> Self {
        Self { line: 0, column_start: 0, column_end: 0 }
    }
}

impl Location {
    fn new(line: usize, column_start: usize, column_end: usize) -> Self {
        Self { line, column_start, column_end }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    IntLiteral(Int),
    FloatLiteral(Float),
    StringLiteral(String),
    Ident(String),
    Keyword(String),
    Punct(char),
    Newline,
    Eof,
}

impl TokenType {
    fn from_str(text: &str) -> Option<Self> {
        let mut chars = text.chars();
        let first_letter = chars.clone().collect::<Vec<char>>()[0];
        Some(if let Ok(val) = text.parse::<Int>() {
            Self::IntLiteral(val)
        }
        else if let Ok(val) = text.parse::<Float>() {
            Self::FloatLiteral(val)
        }
        else if text == "\n" {
            Self::Newline
        }
        else if chars.any(char::is_alphanumeric) && !first_letter.is_numeric() && first_letter != '_' {
            Self::Ident(text.to_string())
        }
        else {
            return None;
        })
    }

    fn is_punct(&self, punct: char) -> bool {
        match self {
            Self::Punct(p) => *p == punct,
            _ => false
        }
    }

    fn is_int(&self) -> Option<i32> {
        match self {
            Self::IntLiteral(i) => Some(*i),
            _ => None
        }
    }

    fn is_ident(&self) -> Option<&str> {
        match self {
            Self::Ident(ident) => Some(ident.as_str()),
            _ => None
        }
    }

    fn is_keyword(&self) -> Option<&str> {
        match self {
            Self::Keyword(kword) => Some(kword.as_str()),
            _ => None,
        }
    }
}

fn pos_of(tokens: &[Token], token: TokenType) -> ParseResult<usize> {
    tokens
        .iter()
        .position(|t| t.ty == token)
        .ok_or(ParseError::from( &tokens[0], ParseErrorTy::ExpectedTokenType(token.clone()) ))
}

fn get_inside_block(tokens: &[Token]) -> ParseResult<&[Token]> {
    assert!(tokens.len() != 0, "tried to parse block from nothing");
    if !tokens[0].ty.is_punct('{') { 
        return Err(ParseError::from(&tokens[0], ParseErrorTy::ExpectedPunct('{')));
    };
    let end = pos_of(tokens, TokenType::Punct('}'))?;
    return Ok(&tokens[1..end]);
}

struct Literal;

#[derive(Debug)]
struct Ident {
    name: String,
}

impl Ident {
    fn new(s: impl ToString) -> Self {
        Self {
            name: s.to_string(),
        }
    }
}

#[derive(Debug)]
enum ValueType {
    Expr(Expr),
    IntLiteral(Int),
}

/// 'Value' is like 'assignee', what's on the right
/// hand side of an assignment
#[derive(Debug)]
struct Value {
    ty: Type,
    value: ValueType,
}

impl Value {
    fn from(ty: Type, tokens: &[Token]) -> ParseResult<(usize, Self)> {
        let n;
        let ret;
        assert!(tokens.len() != 0, "Cannot parse value with no tokens");

        if matches!(ty, Type::Inferred) {
            let (taken, ty) = Self::infer_type(tokens)?;
            let (val_took, val) = Value::from(ty, &tokens[taken..])?;
            n = val_took + taken;
            return Ok((n, val));
        }

        match ty {
            Type::Function {..} => {
                let (taken, expr) = Expr::parse(tokens)?;
                ret = ValueType::Expr(expr);
                n = taken;
            }
            Type::IntLiteral => {
                let int = tokens[0].ty.is_int().ok_or(ParseError::from(&tokens[0], ParseErrorTy::ExpectedLiteralOfType(ty.clone(), tokens[0].ty.clone())));
                let end = pos_of(tokens, TokenType::Punct(';'))?;
                if end != 1 {
                    todo!("Parse maths to get value of int literal (in fact, maybe centralise the whole number parsing system to not rely on a specific type)");
                }
                ret = ValueType::IntLiteral(int?);
                n = end+1;
            }
            other => todo!("Parse value for type {other:?}"),
        }

        Ok((n, Value { ty, value: ret }))
    }

    /// This function will consume tokens if the value is like:
    /// Type { body } or fn() -> ty { body }
    /// Otherwise, it will simply infer and consume nothing
    fn infer_type(tokens: &[Token]) -> ParseResult<(usize, Type)> {
        let n;
        let ret;
        assert!(tokens.len() != 0, "Cannot infer type from no tokens");
        if let Some(kword) = tokens[0].ty.is_keyword() {
            match kword {
                "fn" => {
                    let _start = pos_of(tokens, TokenType::Punct('('))?;
                    let end = pos_of(tokens, TokenType::Punct(')'))?;
                    let block_start = pos_of(tokens, TokenType::Punct('{'))?;

                    // FIXME: bounds checking
                    let args_types = if end == 2 { vec![] } else { Type::parse_args(&tokens[2..end])? };
                    let returns = 
                        if tokens.len() > end + 2 && tokens[end+1].ty.is_punct('-') && tokens[end+2].ty.is_punct('>') {
                            Type::from(&tokens[end+3..block_start])?.1
                        } else {
                            Type::Unit
                        };

                    n = block_start;
                    ret = Type::Function {
                        args: args_types,
                        returns: returns.wrap(),
                    };
                },
                kw => todo!("Implement keyword {kw}"),
            }
        }
        else if let Some(ident) = tokens[0].ty.is_ident() {
            todo!("Infer type from an ident (presumably cannot be done at this stage of parsing)");
            // TODO: InferDefer type for inferring later
        }
        else if let Some(_) = tokens[0].ty.is_int() {
            let end = pos_of(tokens, TokenType::Punct(';'))?;

            if end == 1 {
                n = 0;
                ret = Type::IntLiteral;
            }
            else {
                todo!("Parse maths to infer type of mathematical operation: {tokens:?}");
            }
        }
        else {
            todo!("Infer type from {tokens:?}");
        }
        Ok((n, ret))
    }
}

#[derive(Debug, Clone)]
enum Type {
    Function {
        args: Vec<Type>,
        returns: Box<Type>,
    },
    IntLiteral,
    FloatLiteral,
    Unit,
    Inferred,
}

impl Type {
    /// does not include in number of taken tokens
    /// the closing ':'
    fn from(tokens: &[Token]) -> ParseResult<(usize, Self)> {
        let n;
        let ret;
        // dbg!(&tokens);
        let next_colon = pos_of(tokens, TokenType::Punct(':'))?;
        if next_colon == 0 {
            n = 0;
            ret = Self::Inferred;
        }
        else if next_colon == 1 {
            n = 1;
            ret = Self::parse_simple(&tokens[0])?;
        }
        else {
            todo!("Implement complex types: {tokens:?}")
        };
        Ok((n, ret))
    }

    fn parse_args(tokens: &[Token]) -> ParseResult<Vec<Self>> {
        let mut tys = vec![];
        let mut i = 0;
        let mut s = 0; // TODO: rename this shitass name, already forgot what it means
        while let Some(tok) = tokens.get(i) {
            if matches!(tok.ty, TokenType::Punct(',')) {
                tys.push(Type::from(&tokens[s..i])?.1);
                i += 1;
                s = i;
            }
            else {
                i += 1;
            }
        }
        if s != i {
            tys.push(Type::from(&tokens[s..i])?.1);
        }
        return Ok(tys);
    }

    fn parse_simple(token: &Token) -> ParseResult<Self> {
        match &token.ty {
            TokenType::Ident(i) => {
                Ok(match i.as_str() {
                    "int" => Self::IntLiteral,
                    "float" => Self::FloatLiteral,
                    _ => panic!("Unknown type: {i}"),
                })
            }
            // ERROR: a type can't be a number or string
            _ => return Err(ParseError::from( token, ParseErrorTy::CannotParseTypeFrom(token.ty.clone()) )),
        }
    }

    /// Returns None when called on 'Inferred' or 'DeferInferred'
    fn expects_block(&self) -> Option<bool> {
        match self {
            Self::Function {..} => true,
            Self::IntLiteral | Self::Unit | Self::FloatLiteral => false,
            Self::Inferred => return None,
        }.some()
    }
}

#[derive(Debug)]
enum ExprTy {
    /// A declaration is anything like:
    /// ident : <type>? : expr
    Declaration {
        ident: Ident,
        value: Box<Value>,
    },
    Block {
        exprs: Vec<Expr>,
    },
    Eof,
}

#[derive(Debug)]
struct Expr {
    ty: ExprTy,
    loc: Location,
}

impl Expr {
    fn parse<'a>(tokens: &'a [Token]) -> ParseResult<(usize, Self)> {
        let n; // just so i don't forget
        let expr;
        let mut loc = Location::default();
        if tokens.is_empty() {
            return Err(ParseError { loc: Location::default(), err: ParseErrorTy::EmptyTokens });
        }

        loc.line = tokens[0].loc.line;
        loc.column_start = tokens[0].loc.column_start;

        if let Some(ident) = tokens[0].ty.is_ident() {
            // looking for syntax like:
            // ident : type? : value ;?
            let mut index = 1;
            let ident = Ident::new(ident);

            // dbg!(&tokens[0..5]);
            if tokens[index].ty.is_punct(':') {
                index += 1; // take first colon
                // dbg!(&tokens[index..index+3]);
                let (taken_ty, ty) = Type::from(&tokens[index..])?;
                index += taken_ty;

                if tokens[index].ty.is_punct(':') {
                    index += 1; // take second colon
                    let (taken_val, value) = Value::from(ty, &tokens[index..])?;
                    index += taken_val;

                    n = index;
                    expr = Expr {
                        ty: ExprTy::Declaration {
                            ident,
                            value: value.wrap(),
                        },
                        loc,
                    };
                    println!("Parsing declaration, length: {n} (type: {taken_ty}, value: {taken_val})\nExpr: {expr:#?}");
                    return Ok((n, expr));
                }
            }
        }
        else if let TokenType::Punct('{') = tokens[0].ty {
            let mut inner_tokens = get_inside_block(tokens)?;
            if inner_tokens.is_empty() {
                n = 2;
                expr = Expr {
                    ty: ExprTy::Block {
                        exprs: vec![],
                    },
                    loc,
                };
                return Ok((n, expr));
            }
            else {
                let mut exprs = vec![];
                let mut taken = 0;

                while let Ok((n, expr)) = Expr::parse(inner_tokens) {
                    exprs.push(expr);
                    inner_tokens = &inner_tokens[n..];
                    taken += n;
                }

                n = 2 + taken;
                expr = Expr {
                    ty: ExprTy::Block {
                        exprs,
                    },
                    loc,
                };
                println!("GOOP!");
                return Ok((n, expr));
            }
        }
        else if matches!(tokens[0].ty, TokenType::Eof) {
            println!("Reached end of parsing expressions.");
            return Ok((1, Expr { ty: ExprTy::Eof, loc: loc }));
        }

        if tokens.len() < 4 {
            todo!("Implement other types of expressions: {tokens:?}");
        }

        return Err(ParseError::from(&tokens[0], ParseErrorTy::CannotParse(tokens.to_vec())));
    }
}

pub struct Parser {

}

impl Parser {
    pub fn parse(input: impl ToString) -> ParseResult<()> {
        let input = input.to_string();
        let ast = Ast {
            puncts: PUNCTS,
            keywords: &["fn"],
            string_enter: '\'',
            string_exit: '\'',
            keep_newlines: false,
        };
        let Some(tokens_vec) = ast.parse(&input) else {
            println!("Parsing error, presumably unknown token.");
            return Ok(());
        };
        println!("Input: {input}");
        println!("Tokens: {tokens_vec:#?}");

        let mut exprs: Vec<Expr> = vec![];

        let mut tokens = tokens_vec.as_slice();

        while let Ok((n, expr)) = Expr::parse(tokens) {
            exprs.push(expr);
            tokens = &tokens[n..];
        }
        if let Err(ParseError { err: ParseErrorTy::EmptyTokens, .. }) = Expr::parse(tokens) {
            dbg!(exprs);
            return Ok(());
        }
        else {
            let e = Expr::parse(tokens).err().unwrap();
            return Err(e);
        }
    }
}