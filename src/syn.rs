#![allow(dead_code)]
use crate::tokeniser::*;
use crate::KWORDS;
use std::ops::Index;
use std::fmt::{self, Display};

type Ident = String;
type SynResult<T> = Result<T, SyntaxError>;

trait Optionalise {
	fn to_option(self) -> Option<()>;
}

impl Optionalise for bool {
	fn to_option(self) -> Option<()> {
		match self {
			true => Some(()),
			false => None,
		}
	}
}

macro_rules! todoat {
	($l:expr) => {{
		let _: &Location = & $l;
		panic!("{}: Unfinished code.", $l)
	}};
	($l:expr, $msg:literal $(, $rest:tt)?) => {{
		let _: &Location = & $l;
		panic!(
			concat!("{}: not yet implemented: ", $msg), 
			$l, 
			$($rest)*
		)
	}}
}

macro_rules! panicat {
	($l:expr) => {{
		let _: &Location = & $l;
		panic!("{}: explicit panic.", $l)
	}};
	($l:expr, $msg:literal) => {{
		let _: &Location = & $l;
		panic!(
			concat!("{}: ", $msg), 
			$l
		)
	}};
	($l:expr, $msg:literal, $($rest:tt)*) => {{
		let _: &Location = & $l;
		panic!(
			concat!("{}: ", $msg), 
			$l,
			$($rest)*
		)
	}}
}

#[derive(Debug)]
enum SyntaxError {
	Expected { expect: Vec<String>, got: Token },
	InvalidIdent(Token),
	ReservedIdent(Token),
	EndOfTokens,
}

impl Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
        	Self::Expected { expect, got } => {
        		let len = expect.len();
        		write!(
        			f, 
        			"{loc}: Expected {one_of}{expect}, not {got}.",
        			loc = got.loc,
        			one_of = if expect.len() > 1 { "one of " } else { "" },
        			expect = expect.iter().enumerate()
        				.fold(String::new(), |mut acc, (i, exp)| {
        					acc.push_str(&format!("{}", exp));
        					if i != expect.len() {
        						acc.push_str(", ");
        					}
        					acc
        				}),
        		)
        	}
        	Self::EndOfTokens => write!(f, "Unexpectedly reached the end of tokens while building syntax tree."),
        	Self::InvalidIdent(_token) => todo!(),
        	Self::ReservedIdent(_token) => todo!(),
        }
    }
}

fn is_reserved(ident: &str) -> bool {
	KWORDS.contains(&ident) || Primitive::from_str(ident).is_some() || match ident {
		"true" | "false" => true,
		_ => false
	}
} 

// 'inner outlives 'outer
#[derive(Debug, Clone, Copy)]
struct TokenSlice<'a> {
	tokens: &'a [Token],
	loc: usize,
}

impl<'outer, 'inner: 'outer> TokenSlice<'inner> {
	fn next(&'outer mut self) -> SynResult<&'inner Token> {
		let r = self.tokens.get(self.loc);
		self.loc += 1;
		r.ok_or(SyntaxError::EndOfTokens)
	}

	fn expect(&self, s: &str) -> bool {
		let token = self.tokens.get(0).ok_or(SyntaxError::EndOfTokens).unwrap();
		let tokcond = if let Some(stok) = Token::from_str(s) { stok == *token  } else { false };
		token.is_punct(s)
		|| token.is_keyword(s)
		|| tokcond
	}

	fn next_expect(&'outer mut self, s: &str) -> SynResult<()> {
		let next = self.next()?;
		expect_token(next, s)
	}

	fn next_expect_from(&'outer mut self, strs: &[&str]) -> SynResult<()> {
		let next = self.next()?;
		for s in strs {
			expect_token(next, s)?;
		}
		Ok(())
	}
}

fn expect_token(token: &Token, s: &str) -> SynResult<()> {
	let tokcond = if let Some(stok) = Token::from_str(s) { stok == *token } else { false };
	(token.is_punct(s)
	|| token.is_keyword(s)
	|| tokcond).to_option().ok_or(SyntaxError::Expected {
		expect: vec![s.to_owned()],
		got: token.clone(),
	})
}

impl<'a> Index<usize> for TokenSlice<'a> {
    type Output = Token;
    fn index<'b>(&'b self, i: usize) -> &'b Self::Output {
        &self.tokens[i]
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
	Block(Block),
	Literal(Literal),
	FuncCall(FuncCall),
}

#[derive(Debug, Clone, PartialEq)]
enum Statement {
	Assign(Assign),
	// untyped assignment (functions, structs, enums)
	Declare {
		ident: Ident,
		object: Object,
	},
	FuncCall(FuncCall),
	Return(Type, Expr),
}

#[derive(Debug, Clone, PartialEq)]
struct FuncCall {
	ident: Ident,
	passed: FuncCallParams,
}

#[derive(Debug, Clone, PartialEq)]
enum FuncCallParams {
	// TODO: named
	Unnamed(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
struct Field {
	name: Ident,
	ty: Type,
}

impl Field {
	fn new(name: &str, ty: Type) -> Self {
		Self {
			name: name.to_owned(),
			ty
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
enum Object {
	Function(Function),
	Tuple(Vec<Type>),
	Struct(Struct),
}

#[derive(Debug, Clone, PartialEq)]
struct Function {
	ret: Type,
	args: Struct,
	body: Block,
}

#[derive(Debug, Clone, PartialEq)]
struct Struct {
	fields: Vec<Field>,
}

#[derive(Debug, Clone, PartialEq)]
struct Block(Vec<Statement>);

#[derive(Debug, Clone, PartialEq)]
struct Assign {
	lhs: Pattern,
	ty: Type,
	rhs: Expr,
	kind: Assignment,
}

#[derive(Debug, Clone, PartialEq)]
enum Pattern {
	Ident(Ident),
}

#[derive(Debug, Clone, PartialEq)]
enum Type {
	Primitive(Primitive),
	// TODO:
	Object(Box<Object>),
	Void,
	Infer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Primitive {
	Int,
	Float,
	Bool,
	String,
}

impl Primitive {
	fn from_str(s: &str) -> Option<Self> {
		Some(match s {
			"i32" => Self::Int,
			"f32" => Self::Float,
			"bool" => Self::Bool,
			"string" => Self::String,
			_ => return None,
		})
	}
}

#[derive(Debug, Clone, PartialEq)]
enum Literal {
	Int(i32),
	Float(f32),
	String(String),
	Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
enum Assignment {
	Comptime,
	Mutable,
}

// fn next_token(tokens: &mut Vec<Token>) -> Option<Token> {
// 	// TODO: make this a vecDeque (`veckdeck`)
// 	(tokens.len() > 0).then(|| tokens.remove(0))
// }



// fn get_and_expect_token<'a>(tokens: &'a mut &'a [Token], s: &str) -> Option<()> {
// 	let token = tokens.next()?;
// 	expect_token(&token, s).to_option()
// }

fn parse_ident(next: &Token, tokens: &TokenSlice) -> SynResult<Ident> {
	if let Some(ident) = next.get_ident() {
		if tokens[0].is_punct(",") {
			todoat!(tokens[0].loc, "Parsing patterns is not yet supported, only single identifiers");
		}
		else {
			let r = Ok(ident.to_owned());
			return r;
		}
	}
	Err(SyntaxError::InvalidIdent(next.clone()))
}

fn parse_func_call_args(tokens: &mut TokenSlice) -> SynResult<FuncCallParams> {
	if tokens[1].is_punct("=") {
		todoat!(tokens[1].loc, "named parameters in function calls are not supported yet");
	}

	let mut exprs: Vec<Expr> = vec![];

	let ret = loop {
		let og = tokens.loc;

		let next: &Token = tokens.next()?;
		let loc = next.loc;

		if next.is_punct(")") || next.is_punct("}") {
			tokens.loc = og;
			break FuncCallParams::Unnamed(exprs);
		}
		else if next.is_punct(",") {
			continue;
		}
		else if let Ok((_ty, value)) = parse_expr(Some(next), tokens) {
			exprs.push(value)
		}
		else {
			panicat!(loc, "Expected expression in function call parameters, not {:?}", next);
		}
	};

	Ok(ret)
}

fn parse_struct_fields(tokens: &mut TokenSlice) -> SynResult<Struct> {
	let mut fields: Vec<Field> = vec![];

	let r = loop {
		let og = tokens.loc;
		let next = tokens.next()?;
		if let Ok(ident) = parse_ident(&next, tokens) {
			if is_reserved(&ident) { return Err(SyntaxError::ReservedIdent(next.clone())) };

			tokens.next_expect(":")?;

			let Ok(ty) = parse_type(tokens) else {
				panicat!(next.loc, "Failed to parse type of field.");
			};

			fields.push(Field::new(&ident, ty));
		}
		else if next.is_punct(",") {
			continue;
		}
		else if next.is_punct(")") || next.is_punct("}") {
			tokens.loc = og;
			break Struct { fields };
		}
		else {
			panicat!(next.loc, "Expected an identifier when defining fields, not {:?}", next);
		}
	};

	Ok(r)
}

fn parse_type(tokens: &mut TokenSlice) -> SynResult<Type> {
	let next = tokens.next()?;
	if let Some(ident) = next.get_ident() {
		if let Some(primitive) = Primitive::from_str(ident) {
			Ok(Type::Primitive(primitive))
		}
		else {
			todoat!(next.loc, "Parse type name from identifier: `{}`", ident)
		}
	}
	else {
		todoat!(next.loc, "Parse type from: `{:?}`", next);
	}
}

fn parse_expr(current: Option<&Token>, tokens: &mut TokenSlice) -> SynResult<(Type, Expr)> {
	let (ty, rhs);
	let next = if let Some(c) = current { c } else { tokens.next()? };
	if let Some(int) = next.get_int() {
		ty = Type::Primitive(Primitive::Int);
		rhs = Expr::Literal(Literal::Int(int));
	}
	else if let Some(float) = next.get_float() {
		ty = Type::Primitive(Primitive::Float);
		rhs = Expr::Literal(Literal::Float(float));
	}
	else if let Some(string) = next.get_string() {
		ty = Type::Primitive(Primitive::String);
		rhs = Expr::Literal(Literal::String(string));
	}
	else if let Some("true") = next.get_ident() {
		ty = Type::Primitive(Primitive::Bool);
		rhs = Expr::Literal(Literal::Bool(true));
	}
	else if let Some("false") = next.get_ident() {
		ty = Type::Primitive(Primitive::Bool);
		rhs = Expr::Literal(Literal::Bool(false));
	}
	else {
		dbg!(&tokens.tokens[tokens.loc..]);
		dbg!(&next);

		todoat!(next.loc, "parse some random expression as rhs: {:?}", next);
	}
	return Ok((ty, rhs));
}

fn parse_declaration(ident: Ident, tokens: &mut TokenSlice) -> SynResult<Statement> {
	if is_reserved(&ident) {
		// TODO: wrong loc btw
		panicat!(tokens[0].loc, "Cannot assign to reserved name: `{}`", ident);
	}

	let mut next = tokens.next()?;
	let mut punct_loc = next.loc;

	let mut assign_kind = if next.is_punct(":") { Some(Assignment::Comptime) }
	else if next.is_punct("=") { Some(Assignment::Mutable) }
	else { None };

	// TODO: make typing an object an error
	let decl_ty = if next.is_punct(":") || next.is_punct("=") {
		next = tokens.next()?;
		Type::Infer
	}
	else {
		tokens.loc -= 1;
		let r = parse_type(tokens)?;
		tokens.next_expect_from(&["=", ":"])?;
		assign_kind = if next.is_punct(":") { punct_loc = next.loc; Some(Assignment::Comptime) }
		else if next.is_punct("=") { punct_loc = next.loc; Some(Assignment::Mutable) }
		else { panicat!(next.loc, "Could not glean mutability of decl") };
		next = tokens.next()?;
		r
	};

	let assign_kind = assign_kind.unwrap();
	let assert_assign = |at| {
		if assign_kind != at {
			panicat!(punct_loc, "Cannot declare this type mutably, only as a constant.");
		}
	};

	// parse an object declaration (function etc...)
	{
		// NOTE: parsing objects will return with a ConstDeclare
		match next.get_punct() {
			// function
			// FIXME: this does not allow for tuples, make it check for after 
			// parens for the '->'
			Some("(") => {
				// TODO: parse function args
				let args = if !tokens[0].is_punct(")") {
					let r = parse_struct_fields(tokens)?;
					tokens.next_expect(")")?;
					r
				} else {
					let _ = tokens.next()?;
					Struct { fields: vec![] }
				};

				let next = tokens.next()?;
				let return_type = match next.get_punct() {
					Some("->") => {
						let ret = parse_type(tokens)?;
						
						tokens.next_expect("{")?;
						ret
					}
					Some("{") => Type::Void,
					Some(other_punct) => panicat!(next.loc, "`{}` not supported after function def", other_punct), 
					None => panicat!(next.loc, "Function syntax: `ident :: (arg1: ty, ..) -> .. {{}}`"),
				};

				let ret = Statement::Declare {
					ident: ident.to_owned(),
					object: Object::Function(Function {
						ret: return_type,
						args,
						body: parse_block(tokens).expect("Block parse failed")
					})
				};

				assert_assign(Assignment::Comptime);

				return Ok(
					ret
				);
			}
			Some("{") => {
				todoat!(next.loc, "block assign");
			}
			Some(other) => panicat!(next.loc, "`{}` is not correct assignment syntax", other),
			None => (),
		}

		match next.get_keyword() {
			Some("struct") => todoat!(next.loc, "parse struct"),
			Some("enum") => todoat!(next.loc, "parse enum"),
			Some("import") => todoat!(next.loc, "parse import assignment"),
			Some(other) => panicat!(next.loc, "Keyword `{}` cannot be used when assigning an object.", other),
			None => (),
		}
	}

	// anything else

	let (expr_ty, rhs) = parse_expr(Some(&next), tokens)?;

	let ty = match (decl_ty, expr_ty) {
		(Type::Infer, Type::Infer) => Type::Infer,
		(Type::Infer, parsed) 		 => parsed,
		(declared, 		 Type::Infer) => declared,
		(decl, parsed) => if decl == parsed {
			decl
		} else {
			// TODO: be more permissive here (i.e. n : f32 = 5 should work fine)
			panicat!(
				next.loc,
				"Mismatch between type of value and declared type. Declared: {:?}, value: {:?}", 
				decl,
				parsed,
			)
		}
	};

	tokens.next_expect(";")?;

	Ok(
		Statement::Assign(Assign {
			lhs: Pattern::Ident(ident),
			ty,
			rhs,
			kind: assign_kind,
		})
	)
}

fn parse_block(tokens: &mut TokenSlice) -> SynResult<Block> {
	// tokens.next_expect("{").expect("Block expects opening accolade");

	let mut stmts = Vec::new();
	loop {
		let og = tokens.loc;
		let next = tokens.next()?;
		if expect_token(&next, "}").is_ok() {
			break;
		};

		tokens.loc = og;
		let loc = next.loc;
		if let Ok(stmt) = parse_stmt(tokens) {
			stmts.push(stmt);
		}
		else {
			panicat!(loc, "Failed to parse statement");
		}
	}
	Ok(Block(stmts))
}

fn parse_stmt(tokens: &mut TokenSlice) -> SynResult<Statement> {
	let next = tokens.next()?;

	if let Ok(ident) = parse_ident(&next, tokens) {
		let next = tokens.next()?;
		if next.is_punct(":") {
			// ident : <...>
			return parse_declaration(ident.to_owned(), tokens);
		}
		else if next.is_punct("(") {
			// ident ( <...>
			let passed = if !tokens[0].is_punct(")") {
				let r = parse_func_call_args(tokens)?;
				tokens.next_expect(")")?;
				r
			} else {
				let _ = tokens.next()?;
				FuncCallParams::Unnamed(vec![])
			};

			tokens.next_expect(";")
				.unwrap_or_else(|_| panicat!(next.loc, "allow for chaining function call statement"));

			Ok(Statement::FuncCall(FuncCall { ident: ident.to_owned(), passed }))
		}
		else if next.is_punct(",") {
			// ident , <...>
			todoat!(next.loc, "Parse declaration with pattern");
		}
		else {
			panicat!(next.loc, "unexpected identifier {:?}", next);
		}
	}
	else if let Some(kword) = next.get_keyword() {
		match kword {
			"if" | "switch" | "while" | "loop" => todoat!(next.loc, "parse if and stuff statements"),
			"extern" => todoat!(next.loc, "parse naked import statement"),
			"return" => {
				let (ty, expr) = parse_expr(None, tokens)?;
				Ok(Statement::Return(ty, expr))
			}
			kw => panicat!(next.loc, "Keyword `{}` cannot be used as a statement position for now.", kw),
		}
	}
	else {
		panicat!(next.loc, "{:#?}", next);
	}
}

pub fn parse_items(mut tokens: Vec<Token>) -> Option<()> {
	let mut stmts: Vec<Statement> = Vec::new();
	let mut tokens = TokenSlice {
		tokens: tokens.as_slice(),
		loc: 0,
	};
	while !tokens.tokens[tokens.loc].is_eof() {
		stmts.push(parse_stmt(&mut tokens).unwrap());
	}

	dbg!(stmts);

	Some(())
	// match &tokens[0].ty {
	// 	TokenType::IntLiteral(int) 			=> todo!("parse int literals"),
	//     TokenType::FloatLiteral(float)		=> todo!("parse float literals"),
	//     TokenType::StringLiteral(string)	=> todo!("parse string literals"),
	//     TokenType::Ident(ident) 			=> todo!("parse identifiers"),
	//     TokenType::Keyword(kword) 			=> todo!("parse keywords"),
	//     TokenType::Punct(punct) 			=> todo!("parse punctuation"),
	//     TokenType::Newline 					=> todo!("parse newlines"),
	//     TokenType::Eof 						=> (),
	// }
}