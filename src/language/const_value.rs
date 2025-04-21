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
pub enum ConstantValueType {
    Function {
        block: Expr,
    },
    IntLiteral(Int),
}

/// 'ConstantValue' is like 'assignee', what's on the right
/// hand side of an assignment
#[derive(Debug)]
pub struct ConstantValue {
    ty: Type,
    value: ConstantValueType,
}

impl ConstantValue {
    pub fn from(ty: Type, tokens: &[Token]) -> ParseResult<(usize, Self)> {
        let n;
        let ret;
        assert!(tokens.len() != 0, "Cannot parse value with no tokens");

        if matches!(ty, Type::Inferred) {
            let (taken, ty) = Self::infer_type(tokens)?;
            let (val_took, val) = ConstantValue::from(ty, &tokens[taken..])?;
            n = val_took + taken;
            return Ok((n, val));
        }

        match ty {
            Type::Function {..} => {
                let (taken, expr) = Expr::parse(tokens)?;
                ret = ConstantValueType::Function { block: expr };
                n = taken;
            }
            Type::IntLiteral => {
                let int = tokens[0].ty.is_int().ok_or(ParseError::ExpectedLiteralOfType(ty.clone(), tokens[0].ty.clone()).at(tokens[0].loc));
                let end = tokens.pos_of(TokenType::Punct(';'))?;
                if end != 1 {
                    todo!("Parse maths to get value of int literal (in fact, maybe centralise the whole number parsing system to not rely on a specific type)");
                }
                ret = ConstantValueType::IntLiteral(int?);
                n = end+1;
            }
            other => todo!("Parse value for type {other:?}"),
        }

        Ok((n, ConstantValue { ty, value: ret }))
    }

    /// This function will consume tokens if the value is like:
    /// Type { body } or fn() -> ty { body }
    /// Otherwise, it will simply infer and consume nothing
    fn infer_type(tokens: &[Token]) -> ParseResult<(usize, Type)> {
        
    }
}