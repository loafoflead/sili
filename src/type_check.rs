use crate::{todoat, panicat};
use super::syn::*;
use std::collections::HashMap;

pub fn check(stmts: Vec<StatementLoc>) -> Option<()> {
	let _types: HashMap<Ident, Type> = HashMap::new();
	let functions: HashMap<Ident, Function> = HashMap::new();

	let _scope = 0; // global

	for StatementLoc { stmt, from, to } in stmts {
		match stmt {
			Statement::Assignment(Assign { ../*lhs, ty, rhs, kind*/ }) => todoat!(from => to, "implement assignment typecheck"),
			Statement::TypeDeclaration {../* ident, ty */} => todoat!(from => to, "implement typedecl typecheck"),
			Statement::FunctionDeclaration { ident, function: Function { ret, args, body } } => {
				let gleaned_return_ty = if let Some(StatementLoc { stmt, .. }) = body.0.last() {
					match stmt {
						Statement::Return(ty, ..) => ty,
						// if no return stmt, assume void
						_ => {
							if ret == Type::Void {
								&Type::Void
							}
							else {
								panicat!(from, "Function {} is missing return statement.", ident);
							}
						}
					}
				} else {
					// if func has no stmts in body, assume return is void
					if ret == Type::Void {
						&Type::Void
					}
					else {
						panicat!(from, "Function {} is missing return statement.", ident);
					}
				};
				if *gleaned_return_ty != Type::Infer && *gleaned_return_ty != ret {
					panicat!(from, "Type mismatch in type returned from function, got {}, expected {}.", gleaned_return_ty, ret);
				}
				todoat!(from => to, "implement funcdecl typecheck")
			}
			Statement::FuncCall(_) => todoat!(from => to, "implement funccall typecheck"),
			Statement::Return(_ty, _expr) => todoat!(from => to, "implement return typecheck"),
		}
	}

	Some(())
}