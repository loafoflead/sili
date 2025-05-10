use crate::{todoat, panicat};
use super::syn::*;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Debug)]
struct ScopedIdent {
	ident: Ident,
	// 0 is global scope
	scope: usize,
}

impl ScopedIdent {
	fn new_in(ident: Ident, scope: usize) -> Self {
		Self {
			ident, scope
		}
	}
}

#[derive(Debug)]
struct TypeState {
	types: HashMap<ScopedIdent, Type>,
	// TODO: make functions not 'own' their code, 
	// and just be a singature
	// or create a new type enum here, it just 
	// feels weird having the WHOLE block of code in 
	// the type
	functions: HashMap<ScopedIdent, Function>,
	scope: usize,
}

pub fn type_check(stmts: Vec<StatementLoc>) -> Option<()> {
	let mut state = TypeState {
		types: HashMap::new(),
		functions: HashMap::new(),
		scope: 0,
	};
	check(&mut state, stmts.as_slice())
}

pub fn check(state: &mut TypeState, stmts: &[StatementLoc]) -> Option<()> {
	dbg!(&state);
	let cur_scope = state.scope;
	for StatementLoc { stmt, from, to } in stmts {
		match stmt {
			Statement::Assignment(Assign { ../*lhs, ty, rhs, kind*/ }) => todoat!(from => to, "implement assignment typecheck"),
			Statement::TypeDeclaration {../* ident, ty */} => todoat!(from => to, "implement typedecl typecheck"),
			Statement::FunctionDeclaration { ident, function } => {
				let Function { ret, args, body } = &function;
				let gleaned_return_ty = if let Some(StatementLoc { stmt, .. }) = body.0.last() {
					match stmt {
						Statement::Return(ty, ..) => ty,
						// if no return stmt, assume void
						_ => {
							if *ret == Type::Void {
								&Type::Void
							}
							else {
								panicat!(from, "Function {} is missing return statement.", ident);
							}
						}
					}
				} else {
					// if func has no stmts in body, assume return is void
					if *ret == Type::Void {
						&Type::Void
					}
					else {
						panicat!(from, "Function {} is missing return statement.", ident);
					}
				};

				if *gleaned_return_ty != Type::Infer && *gleaned_return_ty != *ret {
					panicat!(from, "Type mismatch in type returned from function, got {}, expected {}.", gleaned_return_ty, ret);
				}

				state.functions.insert(ScopedIdent::new_in(ident.to_owned(), cur_scope), function.clone());

				// FIXME: this is slightly odd?
				state.scope += 1;
				check(state, body.0.as_slice())?;
				state.scope -= 1;
			}
			Statement::FuncCall(_) => todoat!(from => to, "implement funccall typecheck"),
			Statement::Return(_ty, _expr) => todoat!(from => to, "implement return typecheck"),
		}
	}

	Some(())
}