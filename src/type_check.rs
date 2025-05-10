use crate::todoat;
use super::syn::*;
use std::collections::HashMap;

pub fn check(stmts: Vec<StatementLoc>) -> Option<()> {
	let _types: HashMap<Ident, Type> = HashMap::new();
	let _functions: HashMap<Ident, Function> = HashMap::new();

	let _scope = 0; // global

	for StatementLoc { stmt, from, to } in stmts {
		match stmt {
			Statement::Assignment(Assign { ../*lhs, ty, rhs, kind*/ }) => todoat!(from => to, "implement assignment typecheck"),
			Statement::TypeDeclaration {../* ident, ty */} => todoat!(from => to, "implement typedecl typecheck"),
			Statement::FunctionDeclaration {../* ident, function */} => todoat!(from => to, "implement funcdecl typecheck"),
			Statement::FuncCall(_) => todoat!(from => to, "implement funccall typecheck"),
			Statement::Return(_ty, _expr) => todoat!(from => to, "implement return typecheck"),
		}
	}

	Some(())
}