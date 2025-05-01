use crate::tokeniser::*;

type Ident = String;

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

#[derive(Debug, Clone)]
enum Expr {
	Block(Block),
	If(Box<Expr>, Block),
	Literal(Literal),
}

#[derive(Debug, Clone)]
enum Statement {
	ConstAssign(Assign),
	// untyped assignment (functions, structs, enums)
	ConstDeclare {
		ident: Ident,
		object: Object,
	},
	Assign(Assign),
	Return(Type, Expr),
}

#[derive(Debug, Clone)]
enum Object {
	Function(Function),
}

#[derive(Debug, Clone)]
struct Function {
	ret: Type,
	args: Type,
	body: Block,
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
#[derive(Debug, Clone, Copy)]
enum Type {
	Primitive(Primitive),
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
}

fn get_and_expect_token(tokens: &mut Vec<Token>, s: &str) -> Option<bool> {
	let token = next_token(tokens)?;
	Some( expect_token(&token, s) )
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

fn parse_block(tokens: &mut Vec<Token>) -> Option<Block> {
	get_and_expect_token(tokens, "{")?;
	let mut stmts = Vec::new();
	loop {
		let next = tokens.get(0)?;
		if expect_token(next, "}") { 
			let _ = next_token(tokens);
			break 
		};

		if let Some(stmt) = parse_stmt(tokens) {
			stmts.push(stmt);
		}
		else {
			eprintln!("Failed to parse statement");
			return None;
		}
	}
	Some(Block(stmts))
}

fn parse_stmt(tokens: &mut Vec<Token>) -> Option<Statement> {
	let next = next_token(tokens)?;

	if let Some(ident) = next.get_ident() {
		// TODO: parse pattern
		get_and_expect_token(tokens, ":")?;
		// TODO: parse type
		let ty = Type::Infer;
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
					let args = Type::Void;

					get_and_expect_token(tokens, ")")?;
					let next = next_token(tokens)?;
					let return_type = match next.get_punct() {
						Some("->") => {
							let next = next_token(tokens)?;
							match next.get_ident() {
								Some("i32") => Type::Primitive(Primitive::Int),
								Some(other) => todoat!(next.loc, "Parse type `{}`", other),
								None => panic!("Expect return type for function, not {next:?}"),
							}
						}
						Some("{") => Type::Void,
						Some(other_punct) => panic!("`{other_punct}` not supported when declaring a constant."), 
						None => panic!("Function syntax: `ident :: (arg1: ty, ..) -> .. {{}}`"),
					};

					let ret = Statement::ConstDeclare {
						ident: ident.to_owned(),
						object: Object::Function(Function {
							ret: return_type,
							args,
							body: parse_block(tokens)?
						})
					};

					return Some(
						ret
					);
				}
				Some("{") => {
					todoat!(next.loc, "expression assign");
				}
				Some(other) => panic!("`{other}` is not correct assignment syntax"),
				None => (),
			}

			match next.get_keyword() {
				Some("struct") => todoat!(next.loc, "parse struct"),
				Some("enum") => todoat!(next.loc, "parse enum"),
				Some(other) => panic!("Keyword `{other}` cannot be used when assigning an object."),
				None => (),
			}
		}

		// anything else

		let (ty, rhs) = parse_expr(Some(next), tokens)?;

		get_and_expect_token(tokens, ";")?;

		Some(
			Statement::ConstAssign(Assign {
				lhs: Pattern::Ident(ident.to_owned()),
				ty,
				rhs,
			})
		)
	}
	else if let Some(kword) = next.get_keyword() {
		match kword {
			"if" | "switch" | "while" | "loop" => todo!("parse if and stuff statements"),
			"return" => {
				let (ty, expr) = parse_expr(None, tokens)?;
				Some(Statement::Return(ty, expr))
			}
			kw => panic!("Keyword `{kw}` cannot be used in statement position"),
		}
	}
	else {
		panic!("{next:#?}");
		None
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