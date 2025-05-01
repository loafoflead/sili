use std::fmt::{Display, self};

#[derive(Debug)]
pub struct ParseErrorLoc {
    err: ParseError,
    loc: Location,
}

impl Display for ParseErrorLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        if self.loc == Location::default() {
            write!(f, "? : ERROR: {}", self.err)
        }
        else {
            write!(f, "{} : ERROR : {}", self.loc, self.err)
        }
    }
}

#[derive(Debug)]
pub enum ParseError {
    ExpectedTokenType(Vec<Token>, TokenType),
    ExpectedPunct(char),
    EmptyTokens,
    CannotParse(Vec<Token>),
    CannotParseTypeFrom(TokenType),
    CouldNotFindSequence(Vec<Token>),
    // ExpectedLiteralOfType(Type, TokenType),
    UnwantedToken(TokenType),
    UnexpectedKeyword(String, Vec<String>),
}

impl ParseError {
    fn at(self, loc: Location) -> ParseErrorLoc {
        ParseErrorLoc {
            err: self,
            loc,
        }
    }

    fn loc_unknown(self) -> ParseErrorLoc {
        ParseErrorLoc {
            err: self,
            loc: Location::default(),
        }
    }
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::ExpectedTokenType(toks, tok) => write!(f, "Expected token of type: `{tok:?}` in {toks:?}"),
            // Self::ExpectedLiteralOfType(ty, tok) => write!(f, "Expected literal of type: `{ty:?}`, got `{tok:?}`"),
            Self::CannotParseTypeFrom(tokty) => write!(f, "Cannot parse type from token: `{tokty:?}`, expected identifier"),
            Self::CouldNotFindSequence(toks) => write!(f, "Expected sequence of tokens but could not find them: `{toks:?}`"),
            Self::ExpectedPunct(c) => write!(f, "Expected punctuation: `{c}`"),
            Self::UnwantedToken(tok) => write!(f, "Found token of type: `{tok:?}`, though was hoping not to."),
            Self::UnexpectedKeyword(kw, wanted) => write!(f, "Found keyword: `{kw}` when only {wanted:?} was expected."),
            Self::EmptyTokens => write!(f, "Reached end of tokens or was passed an empty list"),
            Self::CannotParse(tokens) => write!(f, "Could not parse tokens: `{tokens:?}`"),
        }
    }
}

type ParseResult<T> = Result<T, ParseErrorLoc>;





pub struct Tokeniser {
    pub puncts:         &'static [&'static str],
    pub keywords:       &'static [&'static str],
    pub string_enter:   char,
    pub string_exit:    char,
    pub keep_newlines:  bool,
}

impl Tokeniser {
    fn sanity_check(&self) -> Option<()> {
        for (i, punct) in self.puncts.iter().enumerate() {
            for (j, punct2) in self.puncts.iter().enumerate() {
                if i != j && punct == punct2 {
                    return None;
                }
            }
        }
        Some(())
    }

    pub fn tokenise(&self, text: impl ToString) -> Option<Vec<Token>> {
        let Some(()) = self.sanity_check() else { 
            eprintln!("puncts inconsistent");
            return None;
        };
        let string = text.to_string();
        dbg!(&string);
        let chars = string.chars().collect::<Vec<char>>();

        let mut line = 1;
        let mut column_og = 1;
        let mut column = 1;

        let mut tokens: Vec<Token> = vec![];

        let mut buf = String::with_capacity(10);
        let mut in_string = false;

        let mut idx = 0;
        'outer: while idx < chars.len() {
            let c = chars[idx];
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
            else if let Some(_) = self.puncts.iter().find(|p| p.starts_with(c)) {
                let mut puncts: Vec<&&str> = self.puncts.iter().filter(|p| p.starts_with(c)).collect();
                puncts.sort_by(|a: &&&str, b: &&&str| a.len().cmp(&b.len()));
                puncts.reverse();
                for punct in puncts {
                    for (i, pc) in punct.chars().enumerate() {
                        if pc != chars[idx + i] {
                            eprintln!("{pc}, {}", chars[idx + i]);
                            continue;
                        }
                        if i == punct.len() - 1 {
                            if !buf.trim().is_empty() {
                                tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
                                buf.clear();
                                column_og = column - 1;
                            }
                            tokens.push(Token::ty(TokenType::Punct(punct.to_string()), line, column_og, column));
                            column_og = column;
                            idx += punct.len();
                            continue 'outer;
                        }
                    }
                }
                eprintln!("TODO: what to do about malformed punctuation: c: {:?}, s..: {:?}", c, &chars[idx..]);
                return None;
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
            idx += 1;
        }
        // TODO: dry i guess
        if !buf.trim().is_empty() {
            tokens.push(Token::new(buf.as_str().trim(), line, column_og, column)?);
            buf.clear();
            column_og = column;
        }
        tokens.push(Token::ty(TokenType::Eof, line, column_og, column));

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

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub ty: TokenType,
    pub loc: Location,
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

    pub fn from_str(text: &str) -> Option<Self> {
        Some(Self {
            ty: TokenType::from_str(text)?,
            loc: Location::default(),
        })
    }

    pub fn is_punct(&self, punct: &str) -> bool {
        self.ty.is_punct(punct)
    }

    pub fn get_punct(&self) -> Option<&str> {
        self.ty.get_punct()
    }

    pub fn get_int(&self) -> Option<i32> {
        self.ty.get_int()
    }

    pub fn get_float(&self) -> Option<f32> {
        self.ty.get_float()
    }

    pub fn get_string(&self) -> Option<String> {
        self.ty.get_string()
    }

    pub fn get_ident(&self) -> Option<&str> {
        self.ty.get_ident()
    }

    pub fn is_keyword(&self, kword: &str) -> bool {
        self.ty.is_keyword(kword)
    }

    pub fn get_keyword(&self) -> Option<&str> {
        self.ty.get_keyword()
    }

    pub fn is_eof(&self) -> bool {
        self.ty.is_eof()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Location {
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

type Int = i32;
type Float = f32;

#[derive(Debug, PartialEq, Clone)]
pub enum TokenType {
    IntLiteral(Int),
    FloatLiteral(Float),
    StringLiteral(String),
    Ident(String),
    Keyword(String),
    Punct(String),
    Newline,
    Eof,
}

impl TokenType {
    pub fn from_str(text: &str) -> Option<Self> {
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

    pub fn is_punct(&self, punct: &str) -> bool {
        match self {
            Self::Punct(p) => *p == punct,
            _ => false
        }
    }

    pub fn get_punct(&self) -> Option<&str> {
        match self {
            Self::Punct(p) => Some(p),
            _ => None
        }
    }

    pub fn get_int(&self) -> Option<i32> {
        match self {
            Self::IntLiteral(i) => Some(*i),
            _ => None
        }
    }

    pub fn get_float(&self) -> Option<f32> {
        match self {
            Self::FloatLiteral(f) => Some(*f),
            _ => None
        }
    }

    pub fn get_string(&self) -> Option<String> {
        match self {
            Self::StringLiteral(s) => Some(s.clone()),
            _ => None
        }
    }

    pub fn get_ident(&self) -> Option<&str> {
        match self {
            Self::Ident(ident) => Some(ident.as_str()),
            _ => None
        }
    }

    pub fn is_keyword(&self, kword: &str) -> bool {
        match self {
            Self::Keyword(kw) => kw == kword,
            _ => false,
        }
    }

    pub fn get_keyword(&self) -> Option<&str> {
        match self {
            Self::Keyword(kw) => Some(kw),
            _ => None,
        }
    }

    fn is_eof(&self) -> bool {
        if let Self::Eof = self { true } else { false }
    }
}

trait TokenOps<'a> {
    fn pos_of(&'a self, token: TokenType) -> ParseResult<usize>;
    fn get_before(&'a self, tok: TokenType) -> ParseResult<&'a [Token]>;
    fn get_after(&'a self, tok: TokenType) -> ParseResult<&'a [Token]>;
    fn get_between(&'a self, tok1: TokenType, tok2: TokenType) -> ParseResult<&'a [Token]>;
    fn get_between_without(&'a self, tok1: TokenType, tok2: TokenType, exclude: &[TokenType]) -> ParseResult<&'a [Token]>;
    fn find_seq(&'a self, pattern: Vec<Token>) -> ParseResult<(usize, usize)>;
}

impl<'a> TokenOps<'a> for &'a [Token] {
    /// Note: includes the found token
    fn pos_of(&'a self, token: TokenType) -> ParseResult<usize> {
        if self.is_empty() { return Err(ParseError::EmptyTokens.loc_unknown()); }
        self
            .iter()
            .position(|t| t.ty == token)
            .ok_or(ParseError::ExpectedTokenType(self.to_vec(), token.clone()).at(self[0].loc))
    }

    fn get_before(&'a self, tok: TokenType) -> ParseResult<&'a [Token]> {
        let tok_loc = self.pos_of(tok)?;
        Ok(&self[..tok_loc])
    }

    fn get_after(&'a self, tok: TokenType) -> ParseResult<&'a [Token]> {
        let tok_loc = self.pos_of(tok)?;
        Ok(&self[tok_loc+1..]) // don't wan'nit
    }

    fn get_between(&'a self, tok1: TokenType, tok2: TokenType) -> ParseResult<&'a [Token]> {
        let strt = self.pos_of(tok1.clone())?;
        let new = &self[strt+1..];
        let end = new.pos_of(tok2.clone())? + strt + 1;
        Ok(&self[strt+1..end])
    }

    fn get_between_without(&'a self, tok1: TokenType, tok2: TokenType, exclude: &[TokenType]) -> ParseResult<&'a [Token]> {
        let strt = self.pos_of(tok1.clone())?;
        let new = &self[strt+1..];
        let end = new.pos_of(tok2.clone())? + strt + 1;
        for i in strt..end {
            let tok = &self[i];
            if exclude.contains(&tok.ty) {
                return Err(ParseError::UnwantedToken(tok.ty.clone()).at(tok.loc));
            }
        }
        Ok(&self[strt+1..end])
    }

    fn find_seq(&'a self, toks: Vec<Token>) -> ParseResult<(usize, usize)> {
        let mut p = 0;
        let mut s = 0;
        for (i, t) in self.iter().enumerate() {
            if *t == toks[p] {
                if p == 0 { s = i }
                p += 1;
            }
            else {
                s = 0;
                p = 0;
            }
            if p == toks.len() {
                return Ok((s, i));
            }
        }
        Err(ParseError::CouldNotFindSequence(toks).at(self[0].loc))
    }
}