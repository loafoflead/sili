use super::*;

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

#[derive(Debug)]
pub enum ValueType {
    
    IntLiteral(Int),
}

/// 'Value' is like 'assignee', what's on the right
/// hand side of an assignment
#[derive(Debug)]
pub struct Value {
    ty: Type,
    value: ValueType,
}

impl Value {
    pub fn from(ty: Type, tokens: &[Token]) -> ParseResult<(usize, Self)> {
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
                let int = tokens[0].ty.is_int().ok_or(ParseError::ExpectedLiteralOfType(ty.clone(), tokens[0].ty.clone()).at(tokens[0].loc));
                let end = tokens.pos_of(TokenType::Punct(';'))?;
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
                    let arg_tokens = tokens.get_between(t!('('), t!(')'))?;
                    let block = tokens.get_between(t!('{'), t!('}'))?;
                    let block_start = tokens.pos_of(t!('{'))?;
                    let args_end = tokens.pos_of(t!(')'))?; // parens + fn keyword

                    // FIXME: bounds checking
                    let args_types = Type::parse_args(arg_tokens)?;
                    let returns = 
                        if block_start - args_end > 2 && tokens[args_end+1].ty.is_punct('-') && tokens[args_end+2].ty.is_punct('>') {
                            Type::from(tokens.get_between(t!('>'), t!('{'))?)?.1
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
            let end = tokens.pos_of(TokenType::Punct(';'))?;

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