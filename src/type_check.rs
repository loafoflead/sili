use crate::{todoat, panicat};
use super::syn::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
struct TypeState {
	types: Vec<HashMap<Ident, Type>>,
	// TODO: make functions not 'own' their code, 
	// and just be a singature
	// or create a new type enum here, it just 
	// feels weird having the WHOLE block of code in 
	// the type
	functions: Vec<HashMap<Ident, FuncSignature>>,
	// TODO: implementation:
	// find a thing, if you can't get its type from the expressions (i.e. they're not in the database)
	// add the ident to a map of needed types, then continue parsing, so that once
	// the needed type is encountered it can be used to solve that missing ident 
	// or when missing an ident, solve for that one asap, which would be slow but maybe not even slower
	variables: Vec<HashMap<Ident, Type>>,
	scope: usize,
}

impl TypeState {
	// get types from the given scope or above
	fn get_type_above(&self, ident: &str, scope: usize) -> Option<&Type> {
		for sc in 0..scope+1 {
			if let Some(t) = self.types.get(sc).expect("Uninitialised scope").get(ident) {
				return Some(t);
			}
		}
		None
	}
	fn get_typeof_var_above(&self, ident: &str, scope: usize) -> Option<&Type> {
		for sc in 0..scope+1 {
			if let Some(t) = self.variables.get(sc).expect("Uninitialised scope").get(ident) {
				return Some(t);
			}
		}
		None
	}

	fn push_scope(&mut self) {
		self.types.push(HashMap::new());
		self.functions.push(HashMap::new());
		self.variables.push(HashMap::new());
	}
}

pub fn type_check(stmts: Vec<StatementLoc>) -> Option<()> {
	let mut state = TypeState {
		types: Vec::new(),
		functions: Vec::new(),
		variables: Vec::new(),
		scope: 0,
	};
	// first round, parse functions and type declarations
	check_typedecls(&mut state, stmts.as_slice());
	verify_explicit_typing(&mut state, stmts.as_slice());
	typecheck_expressions(&mut state, stmts.as_slice());
	dbg!(&state);
	None
}

fn check_typedecls(state: &mut TypeState, stmts: &[StatementLoc]) -> Option<()> {
	let cur_scope = state.scope;
	state.push_scope();
	
	for StatementLoc { stmt, from, to } in stmts.iter() {
		match stmt {
			Statement::TypeDeclaration { ident, ty } => {
				state.types[cur_scope].insert(ident.clone(), Type::Object(Arc::new(ty.clone())));
			}
			Statement::FunctionDeclaration { ident, function } => {
				let Function { signature, body } = &function;
				let FuncSignature { ret, args } = &signature;
				let gleaned_return_ty = match parse_return_type(state, body.0.as_slice(), cur_scope) {
					Some(ty) => ty,
					None => {
						if *ret == Type::Void {
							Type::Void
						}
						else {
							// TODO: make this multiple steps, this just won't work 
							// if the code assumes all the data *may* exist, 
							// we need to do multiple passes: global -> scope local -> etc...
							panicat!(from, "Function `{}` is missing return statement, even though signature indicates a return type of {}.", ident, ret);
						}
					}
				};

				if gleaned_return_ty != Type::Infer && gleaned_return_ty != *ret {
					panicat!(from, "Type mismatch in type returned from function, got {}, expected {}.", gleaned_return_ty, ret);
				}

				state.functions[cur_scope].insert(ident.to_owned(), signature.clone());

				// FIXME: this is slightly odd?
				state.scope += 1;
				check_typedecls(state, body.0.as_slice())?;
				state.scope -= 1;
			}
			Statement::FuncCall(_) | Statement::Assignment {..} | Statement::Return(..) => (),
		}
	}

	Some(())
}

fn push_var_to_scope(state: &mut TypeState, scope: usize, pat: &Pattern, ty: Type) {
	match pat {
		Pattern::Ident(ident) => {
			state.variables[scope].insert(ident.to_owned(), ty);	
		}
	}
}

// expects all structs and functions to be superficially typed
fn verify_explicit_typing(state: &mut TypeState, stmts: &[StatementLoc]) -> Option<()> {
	let cur_scope = state.scope;
	
	for StatementLoc { stmt, from, to } in stmts.iter() {
		match stmt {
			Statement::TypeDeclaration { ident, ty } => {
				match ty {
					Object::TupleStruct(struc) | Object::Struct(struc) => {
						check_struct_fields(state, &struc, cur_scope)?;
					}
					Object::Enum(en) => todo!("??? should simple enums need variant type checks?"),
				}
			}
			Statement::FunctionDeclaration { ident, function } => {
				let Function { signature, body } = &function;
				let FuncSignature { ret, args } = &signature;

				check_struct_fields(state, args, cur_scope)?;

				// FIXME: this is slightly odd?
				state.scope += 1;
				verify_explicit_typing(state, body.0.as_slice())?;
				state.scope -= 1;
			}
			Statement::Assignment( Assign { lhs, ty, rhs, kind } ) => {
				match ty {
					Type::Identifier(ident) => {
						if let None = state.get_type_above(&ident, cur_scope) {
							panicat!(from, "Unknown or inaccessible type in assignment: {}", ident);
							// return None (or an error ig)
						}
						push_var_to_scope(state, cur_scope, lhs, ty.clone());
					}
					Type::Path(_path) => {
						todo!("parse path type in assignment typechecking");
					}
					Type::Infer => (),
					Type::Object(_) | Type::Primitive(_) | Type::Void => {
						push_var_to_scope(state, cur_scope, lhs, ty.clone());
					}
				}
			}
			Statement::FuncCall(_) | Statement::Return(..) => (),
		}
	}

	Some(())
}

fn typecheck_expressions(state: &mut TypeState, stmts: &[StatementLoc]) -> Option<()> {
	let cur_scope = state.scope;
	for StatementLoc { stmt, from, to } in stmts.iter() {
		match stmt {
			Statement::Assignment( Assign { lhs, ty, rhs, kind } ) => {
				if let Some(final_ty) = typecheck_expr(state, rhs, Some(ty), cur_scope) {
					push_var_to_scope(state, cur_scope, lhs, final_ty);
				}
				else {
					panicat!(from, "Could not infer type of expression: {:?}", stmt);
				}
			}
			Statement::FuncCall(_) | Statement::Return(..) => (),
			Statement::TypeDeclaration {..} => {}
			Statement::FunctionDeclaration { ident, function } => {
				let Function { signature: _, body } = &function;
				// let FuncSignature { ret, args } = &signature;

				// FIXME: this is slightly odd?
				state.scope += 1;
				// TODO: make all of these funcs expect a return type (maybe optional)
				typecheck_expressions(state, body.0.as_slice())?;
				state.scope -= 1;
			}
		}
	}

	Some(())
}

fn check_struct_fields(state: &mut TypeState, st: &Struct, cur_scope: usize) -> Option<()> {
	for field in &st.fields {
		if let Type::Identifier(ident) = &field.ty {
			if let None = state.get_type_above(&ident, cur_scope) {
				panic!("Unknown type in struct field: {}", ident);
				// return None (or an error ig)
			}
		}
		else if let Type::Path(path) = &field.ty {
			todo!("parse path type in struct typechecking");
		}
	}
	Some(())
}

fn typeof_expr(state: &TypeState, expr: &Expr, scope: usize) -> Option<Type> {
	match expr {
		Expr::Block(Block(stmts)) => todo!("find return type of block"),
		Expr::Ident(ident) => {
			if let Some(ty) = state.get_typeof_var_above(ident.as_str(), scope) {
				return Some(ty.clone());
			}
			else {
				return None;
			}
		}
		Expr::Binop(_) => todo!("typecheck binop"),
		Expr::Literal(literal) => {
			return Some(literal.ty());
		}
		Expr::FuncCall(FuncCall { ident, .. }) => {
			Some(state.functions[scope].get(ident).unwrap().ret.clone())
		}
	}
}

// none if no type could be found
fn typecheck_expr<'a>(state: &TypeState, expr: &Expr, ty: Option<&Type>, scope: usize) -> Option<Type> {
	let inferred_type = 
		typeof_expr(state, expr, scope);
	match (&ty, &inferred_type) {
		(Some(Type::Identifier(given)), Some(Type::Object(obj))) => {
			if let Type::Object(obj_ident) = state.get_type_above(given, scope).unwrap() {
				if obj_ident == obj {
					return inferred_type;
				}
			}
			panic!("Whoops, provided a type that was not the inferred object. Provided: {}, expected: {:?}", given, obj);
		}
		(None, None) => {
			return None;
		}
		// couldn't infer, but got one
		(Some(given), None) => {
			return None;
		}
		(Some(Type::Infer), Some(inferred)) => {
			return inferred_type;
		}
		(Some(given), Some(inferred)) => {
			if *given == inferred {
				return inferred_type;
			}
			else {
				panic!("Given type `{given}` does not match type of expression in value place: `{}`", inferred);
			}
		}
		(_, _) => todo!("uhuhghgu"),
	}
}

fn parse_return_type<'a>(state: &TypeState, body: &'a [StatementLoc], scope: usize) -> Option<Type> {
	let Some(StatementLoc { from, to, stmt: Statement::Return(ty, expr) }) = body.last() else { return None };

	match ty {
		Type::Infer => {
			return typecheck_expr(state, expr, Some(ty), scope).or(Some(Type::Infer));
		}
		Type::Identifier(ident) => {
			if let Some(ty) = state.get_type_above(ident, scope) {
				todoat!(from => to, "(possibly used for constructor expr) Handle returning of a known type");
			}
		}
		Type::Path(ident) => {
			unreachable!("Type checking return statement should never be given a path");
		}
		known => return Some(known.clone()),
	}
	unreachable!("hi");
}