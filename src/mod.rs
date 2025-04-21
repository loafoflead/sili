use std::fmt::{Display, self};

mod const_value;
use const_value::ConstantValue;

mod value;
use value::Value;

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

macro_rules! t {
    (str $l: literal) => {{
        let _: &str = $l;
        TokenType::StringLiteral($l)
    }};
    (num $l: literal) => {
        TokenType::from_str(stringify!($l))
    };
    ($l: literal) => {
        TokenType::Punct($l)
    };
    ($ts: tt) => {{
        let ast = Ast {
            puncts: PUNCTS,
            keywords: &["fn"],
            string_enter: '\'',
            string_exit: '\'',
            keep_newlines: false,
        };
        let Some(tokens_vec) = ast.parse(stringify!($ts)) else {
            panic!("Parsing error in token macro");
        };

        tokens_vec
    }}
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
        if tokens.is_empty() {
            n = 0;
            ret = Self::Inferred;
        }
        else if tokens.len() == 1 {
            n = 1;
            ret = Self::parse_simple(&tokens[0])?;
        }
        else {
            todo!("Implement complex types: {tokens:?}")
        };
        Ok((n, ret))
    }

    fn parse_args(tokens: &[Token]) -> ParseResult<Vec<Self>> {
        if tokens.is_empty() { return Ok(vec!()); }

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
            _ => return Err(ParseError::CannotParseTypeFrom(token.ty.clone()).at(token.loc)),
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
    /// A const assignment is anything like:
    /// ident : <type>? : const_val
    ConstantAssignment {
        ident: Ident,
        value: Box<ConstantValue>,
    },
    Assignment {
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
            return Err(ParseError::EmptyTokens.loc_unknown());
        }

        loc.line = tokens[0].loc.line;
        loc.column_start = tokens[0].loc.column_start;

        if let TokenType::Punct('{') = tokens[0].ty {
            let mut inner_tokens = tokens.get_between(t!('{'), t!('}'))?;
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

                loop {
                    let ret = Expr::parse(inner_tokens);
                    if let Err(ParseErrorLoc { err: ParseError::EmptyTokens, .. }) = ret {
                        break;
                    }
                    let (n, expr) = ret?;
                    exprs.push(expr);
                    taken += n;
                    inner_tokens = &inner_tokens[n..];
                }

                n = 2 + taken;
                expr = Expr {
                    ty: ExprTy::Block {
                        exprs,
                    },
                    loc,
                };
                return Ok((n, expr));
            }
        }
        // so we don't hit across lines or blocks
        if let Ok(ty) = tokens.get_between_without(t!(':'), t!(':'), &[t!(';'), t!('{'), t!('}')]) {
            let lhs = tokens.get_before(t!(':'))?;
            let after_second_colon = ty.len()+lhs.len()+2;
            let rhs = &tokens[after_second_colon..];

            if let Some(ident) = tokens[0].ty.is_ident() {
                let ident = Ident::new(ident);
                let (_, ty) = Type::from(ty)?;
                let (value_size, value) = ConstantValue::from(ty, rhs)?;

                n = after_second_colon + value_size;
                expr = Expr {
                    ty: ExprTy::Declaration {
                        ident,
                        value: value.wrap(),
                    },
                    loc,
                };
                // println!("Parsing declaration, length: {n} (size of value: {value_size}, size of type+ident: {after_second_colon})\nExpr: {expr:#?}");
                return Ok((n, expr));
            }
            else {
                todo!("implement multiple assignment, possibly abstract 'Assignee' to a struct like value/type");
            }
        }
        if let Ok(ty) = tokens.get_between_without(t!(':'), t!('='), &[t!(';'), t!('{'), t!('}')]) {
            let lhs = tokens.get_before(t!(':'))?;
            let after_equals = ty.len()+lhs.len()+2;
            let rhs = &tokens[after_equals..];

            if let Some(ident) = tokens[0].ty.is_ident() {
                let ident = Ident::new(ident);
                let (_, ty) = Type::from(ty)?;
                let (value_size, value) = Value::from(ty, rhs)?;

                n = after_equals + value_size;
                expr = Expr {
                    ty: ExprTy::Assignment {
                        ident,
                        value: value.wrap(),
                    },
                    loc,
                };
                // println!("Parsing declaration, length: {n} (size of value: {value_size}, size of type+ident: {after_second_colon})\nExpr: {expr:#?}");
                return Ok((n, expr));
            }
            else {
                todo!("implement multiple assignment, possibly abstract 'Assignee' to a struct like value/type");
            }
        }

        
        if matches!(tokens[0].ty, TokenType::Eof) {
            return Ok((1, Expr { ty: ExprTy::Eof, loc: loc }));
        }

        if tokens.len() < 4 {
            todo!("Implement other types of expressions: {tokens:?}");
        }

        return Err(ParseError::CannotParse(tokens.to_vec()).at(tokens[0].loc));
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
        // println!("Tokens: {tokens_vec:#?}");

        let mut exprs: Vec<Expr> = vec![];

        let mut tokens = tokens_vec.as_slice();

        // TODO: iterator mf
        loop {
            let ret = Expr::parse(tokens);
            if let Err(ParseErrorLoc { err: ParseError::EmptyTokens, .. }) = ret {
                break;
            }
            if let Err(e) = ret {
                println!("{}", e);
                break;
            }
            let (n, expr) = ret?;
            exprs.push(expr);
            tokens = &tokens[n..];
        }

        dbg!(exprs);

        Ok(())
    }
}