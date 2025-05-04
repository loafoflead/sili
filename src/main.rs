// use std::io;
// use std::fmt::{self, Display};
// use std::mem;

mod syn;
mod tokeniser;

const PUNCTS: &[&str] = &["(", ")", "{", "}", ";", ":", ",", "->", "<", ">", "-", "+", "{]*=*["];
const KWORDS: &[&str] = &["struct", "enum", "if", "else", "return", "ext"];

fn create_tokeniser() -> tokeniser::Tokeniser {
    tokeniser::Tokeniser {
        puncts: PUNCTS,
        keywords: KWORDS,
        string_enter: '"',
        string_exit: '"',
        keep_newlines: false,
    }
}

#[test] 
fn basic() {
    let tokeniser = create_tokeniser();

    let snippet = 
r#"
value :: 5.5;
name :: "hello";
boolean_f :: false;
boolean_t :: true;

main :: () -> i32 {
    a :: 5;
    b :: 69420;
}
"#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    assert!(syn::parse_items(tokens).is_some());
}

#[test] 
fn function_arguments_and_fields() {
    let tokeniser = create_tokeniser();

    let snippet = 
r#"
foo :: (input: string, height: f32) {}

main :: () -> string {
    a : i32 : 5;
    foo("hello", 7.7);
}
"#;

    let tokens = tokeniser.tokenise(snippet).unwrap();

    assert!(syn::parse_items(tokens).is_some());
}

fn main() {
    let tokeniser = create_tokeniser();
    let snippet = 
r#"
main :: () {
    a :: 5;
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
