use std::fmt::{Display, self};

const PUNCTS: &[char] = &['(', ')', ';', ':'];

struct Ast {
    puncts:         &'static [char],
    keywords:       &'static [&'static str],
    string_enter:   char,
    string_exit:    char,
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
                    tokens.push(Token::new("\n", line, column_og, column)?);
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

impl Location {
    fn new(line: usize, column_start: usize, column_end: usize) -> Self {
        Self { line, column_start, column_end }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    IntLiteral(i32),
    FloatLiteral(f32),
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
        Some(if let Ok(val) = text.parse::<i32>() {
            Self::IntLiteral(val)
        }
        else if let Ok(val) = text.parse::<f32>() {
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
}

struct Ident {
    name: String,
}

enum Type {
    Function {
        args: Vec<Type>,
        return: Box<Type>,
    },
    Unit,
}

enum ExprTy {
    Def {
        id: Ident,
        ty: Type,
    }
}

struct Expr {
    ty: ExprTy,
    loc: Location,
}

impl Expr {
    fn def(id: impl ToString, ty: Type, loc: Location) -> Self {

    }
}

pub struct Parser {

}

impl Parser {
    pub fn parse(input: impl ToString) -> Option<()> {
        let input = input.to_string();
        let ast = Ast {
            puncts: PUNCTS,
            keywords: &[],
            string_enter: '\'',
            string_exit: '\'',
        };
        let Some(tokens) = ast.parse(&input) else {
            println!("Parsing error, presumably unknown token.");
            return None;
        };
        println!("Input: {input}");
        println!("Tokens: {tokens:#?}");

        let mut exprs = vec![];

        let mut i = 0;
        while let Some(tok) = tokens.get(i) {
            let mut n = 1;
            let loc = tok.loc.clone();
            match &tok.ty {
                TokenType::Ident(kword) => {
                    
                }
                any => todo!("TokenType: {any:?}"),
            }
            i += n;
        }

        Some(())
    }
}