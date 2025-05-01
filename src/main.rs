// use std::io;
// use std::fmt::{self, Display};
// use std::mem;

mod syn;
mod tokeniser;

const PUNCTS: &[&str] = &["(", ")", "{", "}", ";", ":", ",", "->", "<", ">", "-", "+", "{]*=*["];
const KWORDS: &[&str] = &["struct", "enum", "if", "else", "return"];

fn main() {
    let tokeniser = tokeniser::Tokeniser {
        puncts: PUNCTS,
        keywords: KWORDS,
        string_enter: '"',
        string_exit: '"',
        keep_newlines: false,
    };

    let _works = 
r#"
value :: 5.5;
name :: "hello";
boolean :: false;
boolean :: true;

main :: () -> i32 {
    a :: 5;
    b :: 69420;
}
"#;

    let snippet = 
r#"
foo :: () {}

main :: () -> i32 {
    a :: 5;
    foo();
}
"#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    let _items = syn::parse_items(tokens);
}

/*
fn main() {
    language::Parser::parse(
// main : fn() : {
//     print 'hi';
// }
// main :: fn() {
"
    CONST :: 12303;
    CONST :: 12303;
VAL :int: 122;
"
// }
// Expr::Declaration {
//   ident: "main",
//   ty: Type::Function { args: [], returns: Unit },
//   value: Value::Expr {
//       expr: Expr::Block {..},
//   } 
// }
// Expr::Declaration {
//   ident: "val",
//   ty: Type::IntLiteral,
//   value: Value::Int(1),
// }
// "main :: fn() {
//     print 'hi';
// }"
).unwrap();
}
*/