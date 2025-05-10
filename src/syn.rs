#![allow(dead_code)]
use crate::tokeniser::*;
use crate::KWORDS;
use std::ops::Index;
use std::fmt::{self, Display};

pub type Ident = String;
type SynResult<T> = Result<T, SyntaxErrorLoc>;

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

#[macro_export]
macro_rules! todoat {
	($l:expr) => {{
		let _: &crate::tokeniser::Location = & $l;
		panic!("{}: Unfinished code.", $l)
	}};
	($l:expr, $msg:literal $(, $rest:tt)?) => {{
		let _: &crate::tokeniser::Location = & $l;
		panic!(
			concat!("{}: not yet implemented: ", $msg), 
			$l, 
			$($rest)*
		)
	}};
	($l:ident => $l2:ident, $msg:literal $(, $rest:tt)?) => {{
		let _: (&crate::tokeniser::Location, &crate::tokeniser::Location) = (& $l, & $l2);
		// TODO: render this from to business
		let _ = $l2; 
		panic!(
			concat!("{}: not yet implemented: ", $msg), 
			$l,
			$($rest)*
		)
	}}
}

#[macro_export]
macro_rules! panicat {
	($l:expr) => {{
		let _: &crate::tokeniser::Location = & $l;
		panic!("{}: explicit panic.", $l)
	}};
	($l:expr, $msg:literal) => {{
		let _: &crate::tokeniser::Location = & $l;
		panic!(
			concat!("{}: ", $msg), 
			$l
		)
	}};
	($l:expr, $msg:literal, $($rest:tt)*) => {{
		let _: &crate::tokeniser::Location = & $l;
		panic!(
			concat!("{}: ", $msg), 
			$l,
			$($rest)*
		)
	}}
}

#[derive(Debug)]
pub struct SyntaxErrorLoc {
	pub loc: Option<Location>,
	pub err: SyntaxError,
}

impl Display for SyntaxErrorLoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
    	if let Some(loc) = self.loc {
    		write!(f, "{}: {}", loc, self.err)
    	}
    	else {
    		write!(f, "[??]: {}", self.err)
    	}
    }
}

#[derive(Debug)]
pub enum SyntaxError {
	Expected { expect: Vec<String>, got: Token },
	InvalidIdent(Token),
	ReservedIdent(Token),
	MalformedPath(Token),
	EndOfTokens,
}

impl SyntaxError {
	fn at(self, loc: Location) -> SyntaxErrorLoc {
		SyntaxErrorLoc { loc: Some(loc), err: self }
	}

	fn at_token_or_default(self, tok: Option<&Token>) -> SyntaxErrorLoc {
		let loc = if let Some(tok) = tok {
			Some(tok.loc)
		} else { None };
		SyntaxErrorLoc { loc, err: self }
	}

	fn somewhere(self) -> SyntaxErrorLoc {
		SyntaxErrorLoc { loc: None, err: self }
	}
}

impl Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
        	Self::Expected { expect, got } => {
        		write!(
        			f, 
        			"Expected {one_of}{expect}, not {got}.",
        			one_of = if expect.len() > 1 { "one of " } else { "" },
        			expect = expect.iter().enumerate()
        				.fold(String::new(), |mut acc, (i, exp)| {
        					acc.push_str(exp);
        					if i != expect.len()-1 {
        						acc.push_str(", ");
        					}
        					acc
        				}),
        		)
        	}
        	Self::EndOfTokens => write!(f, "Unexpectedly reached the end of tokens while building syntax tree."),
        	Self::InvalidIdent(_token) => todo!(),
        	Self::MalformedPath(token) => write!(f, "Path wasn't expecting a {}", token),
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
		r.ok_or(SyntaxError::EndOfTokens.at_token_or_default(r))
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
			if let Ok(_) = expect_token(next, s) {
				return Ok(())
			}
		}
		Err(SyntaxError::Expected {
			expect: strs.iter().map(|s| s.to_string()).collect(),
			got: next.clone(),
		}.at(next.loc))
	}

	fn current_loc(&self) -> SynResult<Location> {
		let token = self.tokens.get(0).ok_or(SyntaxError::EndOfTokens.somewhere())?;
		Ok(token.loc)
	}
}

fn expect_token(token: &Token, s: &str) -> SynResult<()> {
	let tokcond = if let Some(stok) = Token::from_str(s) { stok == *token } else { false };
	(token.is_punct(s)
	|| token.is_keyword(s)
	|| tokcond).to_option().ok_or(SyntaxError::Expected {
		expect: vec![s.to_owned()],
		got: token.clone(),
	}.at(token.loc))
}

fn expect_token_from(token: &Token, strs: &[&str]) -> SynResult<()> {
	for s in strs {
		if let Ok(_) = expect_token(token, s) {
			return Ok(())
		}
	}
	Err(SyntaxError::Expected {
		expect: strs.iter().map(|s| s.to_string()).collect(),
		got: token.clone(),
	}.at(token.loc))
}


impl<'a> Index<usize> for TokenSlice<'a> {
    type Output = Token;
    fn index<'b>(&'b self, i: usize) -> &'b Self::Output {
        &self.tokens[i]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatementLoc {
	pub stmt: Statement,
	pub from: Location,
	pub to: Location,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
	Block(Block),
	Ident(Ident),
	Literal(Literal),
	FuncCall(FuncCall),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
	Assignment(Assign),
	// untyped assignment (functions, structs, enums)
	TypeDeclaration {
		ident: Ident,
		ty: Object,
	},
	FunctionDeclaration {
		ident: Ident,
		function: Function,
	},
	FuncCall(FuncCall),
	Return(Type, Expr),
}

impl Statement {
	fn between(self, from: Location, to: Location) -> StatementLoc {
		StatementLoc {
			stmt: self,
			from,
			to,
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncCall {
	pub ident: Ident,
	pub passed: FuncCallParams,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FuncCallParams {
	// TODO: named
	Unnamed(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
	pub name: Ident,
	pub ty: Type,
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
pub enum Object {
	Tuple(Vec<Type>),
	Struct(Struct),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
	pub ret: Type,
	pub args: Struct,
	pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Struct {
	pub fields: Vec<Field>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block(pub Vec<StatementLoc>);

#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
	pub lhs: Pattern,
	pub ty: Type,
	pub rhs: Expr,
	pub kind: Assignment,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
	Ident(Ident),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
	Primitive(Primitive),
	// TODO:
	Object(Box<Object>),
	Void,
	// Infer used when we have an expression
	// but we can't yet figure out it's type
	// like 'return a', or
	// 'a :: Type<i32>;'
	Infer,
	Identifier(Ident),
	Path(Vec<Ident>),
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
        	Type::Primitive(p) => write!(f, "primitive({p:?})"),
        	Type::Object(obj) => todo!("format obj"),
        	Type::Void => write!(f, "[void]"),
        	Type::Infer => write!(f, "[inferred]"),
        	Type::Identifier(ident) => write!(f, "ident({ident})"),
        	Type::Path(path) => write!(f, "path({})", path.join("::")),
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive {
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
pub enum Literal {
	Int(i32),
	Float(f32),
	String(String),
	Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Assignment {
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
	Err(SyntaxError::InvalidIdent(next.clone()).at(next.loc))
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
			if is_reserved(&ident) { return Err(SyntaxError::ReservedIdent(next.clone()).at(next.loc)) };

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
			return Err(SyntaxError::Expected { expect: vec!["identifier".to_string()], got: next.clone() }.at(next.loc));
			// panicat!(next.loc, "Expected an identifier when defining fields, not {:?}", next);
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
			let loc = tokens.loc;
			let mut next = tokens.next()?;
			if next.is_punct("<") {
				todoat!(next.loc, "Parse generics: `{:?}`", next);
			}
			else if next.is_punct(".") {
				todoat!(next.loc, "Parse paths: `{:?}`", next);

				// let mut path = vec![ident.to_owned()];
				// while next.is_punct(".") {
				// 	next = tokens.next()?;
				// 	if let Some(ident) = next.get_ident() {
				// 		path.push(ident.to_string());
				// 	}
				// 	else {
				// 		return Err(SyntaxError::MalformedPath(next.clone()).at(next.loc));
				// 	}
				// 	next = tokens.next()?;
				// }
				// Ok(Type::Path(path))
			}
			else {
				tokens.loc = loc;
				Ok(Type::Identifier(ident.to_owned()))
			}
		}
	}
	else if next.is_punct("[") {
		todoat!(next.loc, "Parse type of arrays: `{:?}`", next);
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
	else if let Some(ident) = next.get_ident() {
		let loc = tokens.loc;
		let next = tokens.next()?;
		expect_token_from(next, &[";", ")", "}"])
			.unwrap_or_else(|_| panicat!(next.loc, "Parse more complicated expressions"));
		tokens.loc = loc;
		ty = Type::Infer;
		rhs = Expr::Ident(ident.to_owned());
	}
	else {
		dbg!(&tokens.tokens[tokens.loc..]);
		dbg!(&next);

		todoat!(next.loc, "parse some random expression as rhs: {:?}", next);
	}
	return Ok((ty, rhs));
}

fn parse_declaration(ident: Ident, tokens: &mut TokenSlice) -> SynResult<StatementLoc> {
	if is_reserved(&ident) {
		// TODO: wrong loc btw
		panicat!(tokens[0].loc, "Cannot assign to reserved name: `{}`", ident);
	}

	let mut next = tokens.next()?;
	let first_loc = next.loc;
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
		next = tokens.next()?;
		expect_token_from(next, &["=", ":"])?;
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

				let ret = Statement::FunctionDeclaration {
					ident: ident.to_owned(),
					function: Function {
						ret: return_type,
						args,
						body: parse_block(tokens)?
					}
				}.between(first_loc, tokens.current_loc()?);

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
			Some("struct") => {
				tokens.next_expect("{")?;
				let struc = parse_struct_fields(tokens)?;
				tokens.next_expect("}")?;

				assert_assign(Assignment::Comptime);

				return Ok(Statement::TypeDeclaration {
					ident: ident.to_owned(),
					ty: Object::Struct(struc)
				}.between(first_loc, tokens.current_loc()?));
			}
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
		Statement::Assignment(Assign {
			lhs: Pattern::Ident(ident),
			ty,
			rhs,
			kind: assign_kind,
		}).between(first_loc, tokens.current_loc()?)
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
		stmts.push(parse_stmt(tokens)?);
	}
	Ok(Block(stmts))
}

fn parse_stmt(tokens: &mut TokenSlice) -> SynResult<StatementLoc> {
	let next = tokens.next()?;
	let first_loc = next.loc;

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

			Ok(Statement::FuncCall(FuncCall { ident: ident.to_owned(), passed }).between(first_loc, tokens.current_loc()?))
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
				tokens.next_expect(";")?;
				Ok(Statement::Return(ty, expr).between(first_loc, tokens.current_loc()?))
			}
			kw => panicat!(next.loc, "Keyword `{}` cannot be used as a statement position for now.", kw),
		}
	}
	else {
		panicat!(next.loc, "{:#?}", next);
	}
}

pub fn parse_items(snippet: &str, tokens: Vec<Token>) -> Option<Vec<StatementLoc>> {
	let mut stmts: Vec<StatementLoc> = Vec::new();
	let mut tokens = TokenSlice {
		tokens: tokens.as_slice(),
		loc: 0,
	};

	let lines = snippet.split('\n').collect::<Vec<&str>>();

	while !tokens.tokens[tokens.loc].is_eof() {
		let stmt = parse_stmt(&mut tokens);
		if let Err(e) = stmt {
			if let Some(loc) = e.loc {
				println!("{}", lines[loc.line-1]);
				println!("{}{}", " ".repeat(loc.column_start-1), "^".repeat(loc.column_end - loc.column_start));
			}
			println!("{}", e);
			return None;
		}
		else {
			stmts.push(stmt.unwrap());
		}
	}

	Some(stmts)
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