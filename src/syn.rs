#![allow(dead_code)]
use crate::tokeniser::*;

type Ident = String;

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
	($l:expr, $msg:literal $(, $rest:tt)?) => {{
		let _: &Location = & $l;
		panic!(
			concat!("{}: ", $msg), 
			$l, 
			$($rest)*
		)
	}}
}

#[derive(Debug, Clone)]
enum Expr {
	Block(Block),
	Literal(Literal),
	FuncCall(FuncCall),
}

#[derive(Debug, Clone)]
enum Statement {
	ConstAssign(Assign),
	// untyped assignment (functions, structs, enums)
	ConstDeclare {
		ident: Ident,
		object: Object,
	},
	FuncCall(FuncCall),
	Assign(Assign),
	Return(Type, Expr),
}

#[derive(Debug, Clone)]
struct FuncCall {
	ident: Ident,
	passed: FuncCallParams,
}

#[derive(Debug, Clone)]
enum FuncCallParams {
	// TODO: named
	Unnamed(Vec<Expr>),
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
enum Object {
	Function(Function),
	Tuple(Vec<Type>),
	Struct(Struct),
}

#[derive(Debug, Clone)]
struct Function {
	ret: Type,
	args: Struct,
	body: Block,
}

#[derive(Debug, Clone)]
struct Struct {
	fields: Vec<Field>,
}

#[derive(Debug, Clone)]
struct Block(Vec<Statement>);

#[derive(Debug, Clone)]
struct Assign {
	lhs: Pattern,
	ty: Type,
	rhs: Expr,
}

#[derive(Debug, Clone)]
enum Pattern {
	Ident(Ident),
}

#[derive(Debug, Clone)]
enum Type {
	Primitive(Primitive),
	// TODO:
	Object(Box<Object>),
	Void,
	Infer,
}

#[derive(Debug, Clone, Copy)]
enum Primitive {
	Int,
	Float,
	Bool,
	String,
}
#[derive(Debug, Clone)]
enum Literal {
	Int(i32),
	Float(f32),
	String(String),
	Bool(bool),
}

fn next_token(tokens: &mut Vec<Token>) -> Option<Token> {
	// TODO: make this a vecDeque (`veckdeck`)
	(tokens.len() > 0).then(|| tokens.remove(0))
}

fn expect_token(token: &Token, s: &str) -> bool {
	let tokcond = if let Some(stok) = Token::from_str(s) { stok == *token  } else { false };
	token.is_punct(s)
	|| token.is_keyword(s)
	|| tokcond
}

fn get_and_expect_token(tokens: &mut Vec<Token>, s: &str) -> Option<()> {
	let token = next_token(tokens)?;
	expect_token(&token, s).to_option()
}

fn parse_expr(current: Option<Token>, tokens: &mut Vec<Token>) -> Option<(Type, Expr)> {
	let (ty, rhs);
	let next = if let Some(c) = current { c } else { next_token(tokens)? };
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
		todoat!(next.loc, "parse some random expression as rhs: {:?}", next);
	}
	return Some((ty, rhs));
}

fn parse_declaration(ident: Ident, tokens: &mut Vec<Token>) -> Option<Statement> {
	// TODO: parse type
	get_and_expect_token(tokens, ":")?;

	let next = next_token(tokens)?;

	// parse an object declaration (function etc...)
	{
		// NOTE: parsing objects will return with a ConstDeclare
		match next.get_punct() {
			// function
			// FIXME: this does not allow for tuples, make it check for after 
			// parens for the '->'
			Some("(") => {
				// TODO: parse function args
				let args = Struct { fields: vec![] };

				get_and_expect_token(tokens, ")")?;
				let next = next_token(tokens)?;
				let return_type = match next.get_punct() {
					Some("->") => {
						let next = next_token(tokens)?;
						let ret = match next.get_ident() {
							Some("i32") => Type::Primitive(Primitive::Int),
							Some(other) => todoat!(next.loc, "Parse type `{}`", other),
							None => panicat!(next.loc, "Expect return type for function, not {:?}", next),
						};
						get_and_expect_token(tokens, "{")?;
						ret
					}
					Some("{") => Type::Void,
					Some(other_punct) => panicat!(next.loc, "`{}` not supported after function def", other_punct), 
					None => panicat!(next.loc, "Function syntax: `ident :: (arg1: ty, ..) -> .. {{}}`"),
				};

				let ret = Statement::ConstDeclare {
					ident: ident.to_owned(),
					object: Object::Function(Function {
						ret: return_type,
						args,
						body: parse_block(tokens).expect("Block parse failed")
					})
				};

				return Some(
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
			Some(other) => panicat!(next.loc, "Keyword `{}` cannot be used when assigning an object.", other),
			None => (),
		}
	}

	// anything else

	let (ty, rhs) = parse_expr(Some(next), tokens)?;

	get_and_expect_token(tokens, ";")?;

	Some(
		Statement::ConstAssign(Assign {
			lhs: Pattern::Ident(ident),
			ty,
			rhs,
		})
	)
}

fn parse_block(tokens: &mut Vec<Token>) -> Option<Block> {
	// get_and_expect_token(tokens, "{").expect("Block expects opening accolade");

	let mut stmts = Vec::new();
	// dbg!(&tokens);
	loop {
		let next = tokens.get(0)?;
		if expect_token(next, "}") { 
			let _ = next_token(tokens);
			break;
		};

		let loc = next.loc;
		if let Some(stmt) = parse_stmt(tokens) {
			stmts.push(stmt);
		}
		else {
			panicat!(loc, "Failed to parse statement");
		}
	}
	Some(Block(stmts))
}

fn parse_stmt(tokens: &mut Vec<Token>) -> Option<Statement> {
	let next = next_token(tokens)?;

	if let Some(ident) = next.get_ident() {
		let next = next_token(tokens)?;
		if next.is_punct(":") {
			return parse_declaration(ident.to_owned(), tokens);
		}
		else if next.is_punct("(") {
			// TODO: parse args
			get_and_expect_token(tokens, ")")
				.or_else(|| todoat!(next.loc, "parse function arguments"));

			get_and_expect_token(tokens, ";")
				.or_else(|| panicat!(next.loc, "allow for chaining function call statement"));

			Some(Statement::FuncCall(FuncCall { ident: ident.to_owned(), passed: FuncCallParams::Unnamed(vec![]) }))
		}
		else if next.is_punct(",") {
			todoat!(next.loc, "Parse declaration with pattern");
		}
		else {
			panicat!(next.loc, "unexpected identifier.")
		}
	}
	else if let Some(kword) = next.get_keyword() {
		match kword {
			"if" | "switch" | "while" | "loop" => todo!("parse if and stuff statements"),
			"return" => {
				let (ty, expr) = parse_expr(None, tokens)?;
				Some(Statement::Return(ty, expr))
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
	while !tokens[0].is_eof() {
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